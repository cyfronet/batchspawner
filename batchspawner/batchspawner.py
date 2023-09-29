# Copyright (c) Regents of the University of Minnesota
# Copyright (c) Michael Gilbert
# Distributed under the terms of the Modified BSD License.

"""Batch spawners

This file contains an abstraction layer for batch job queueing systems, and implements
Jupyterhub spawners for Torque, SLURM, and eventually others.

Common attributes of batch submission / resource manager environments will include notions of:
  * queue names, resource manager addresses
  * resource limits including runtime, number of processes, memory
  * singleuser child process running on (usually remote) host not known until runtime
  * job submission and monitoring via resource manager utilities
  * remote execution via submission of templated scripts
  * job names instead of PIDs
"""
import asyncio
import pwd
import os
import re
import tempfile
import pathlib
import json
from enum import Enum
from contextlib import contextmanager

from async_generator import async_generator, yield_

from jinja2 import Template

from tornado import gen
from tornado.httpclient import AsyncHTTPClient, HTTPClientError, HTTPRequest

from jupyterhub.spawner import Spawner
from jupyterhub.spawner import set_user_setuid
from jupyterhub.utils import random_port
from traitlets import Integer, Unicode, Float, Dict, default

from cryptography.hazmat.primitives import serialization as crypto_serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend as crypto_default_backend


def format_template(template, *args, **kwargs):
    """Format a template, either using jinja2 or str.format().

    Use jinja2 if the template is a jinja2.Template, or contains '{{' or
    '{%'.  Otherwise, use str.format() for backwards compatability with
    old scripts (but you can't mix them).
    """
    if isinstance(template, Template):
        return template.render(*args, **kwargs)
    if "{{" in template or "{%" in template:
        return Template(template).render(*args, **kwargs)
    return template.format(*args, **kwargs)


def dump_to_rdonly_file(path, content: bytes):
    """Create read-only file and write content to it"""
    flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
    fd = os.open(path, flags=flags, mode=0o400)
    os.write(fd, content)
    os.close(fd)

@contextmanager
def tempkey(key: str, cert: str, login: str):
    """Create temporary directory for key and cert with name based on login"""
    tmpdir = tempfile.TemporaryDirectory(prefix=f'.ssh_{login}_')

    dirpath = pathlib.Path(tmpdir.name)

    keypath = dirpath / 'key'
    dump_to_rdonly_file(keypath, key.encode('utf-8'))
    dump_to_rdonly_file(dirpath / 'key-cert.pub', cert.encode('utf-8'))

    try:
        yield keypath.absolute()
    finally:
        tmpdir.cleanup()


class JobStatus(Enum):
    NOTFOUND = 0
    RUNNING = 1
    PENDING = 2
    UNKNOWN = 3


class BatchSpawnerBase(Spawner):
    """Base class for spawners using resource manager batch job submission mechanisms

    This base class is developed targetting the TorqueSpawner and SlurmSpawner, so by default
    assumes a qsub-like command that reads a script from its stdin for starting jobs,
    a qstat-like command that outputs some data that can be parsed to check if the job is running
    and on what remote node, and a qdel-like command to cancel a job. The goal is to be
    sufficiently general that a broad range of systems can be supported with minimal overrides.

    At minimum, subclasses should provide reasonable defaults for the traits:
        batch_script
        batch_submit_cmd
        batch_query_cmd
        batch_cancel_cmd

    and must provide implementations for the methods:
        state_ispending
        state_isrunning
        state_gethost
    """

    # override default since batch systems typically need longer
    start_timeout = Integer(300).tag(config=True)

    # override default server ip since batch jobs normally running remotely
    ip = Unicode(
        "0.0.0.0",
        help="Address for singleuser server to listen at",
    ).tag(config=True)

    exec_prefix = Unicode(
        "",
        help="Standard executon prefix (e.g. the default sudo -E -u {username})",
    ).tag(config=True)

    # all these req_foo traits will be available as substvars for templated strings
    req_queue = Unicode(
        "",
        help="Queue name to submit job to resource manager",
    ).tag(config=True)

    req_host = Unicode(
        "",
        help="Host name of batch server to submit job to resource manager",
    ).tag(config=True)

    req_memory = Unicode(
        "",
        help="Memory to request from resource manager",
    ).tag(config=True)

    req_nprocs = Unicode(
        "",
        help="Number of processors to request from resource manager",
    ).tag(config=True)

    req_ngpus = Unicode(
        "",
        help="Number of GPUs to request from resource manager",
    ).tag(config=True)

    req_runtime = Unicode(
        "",
        help="Length of time for submitted job to run",
    ).tag(config=True)

    req_partition = Unicode(
        "",
        help="Partition name to submit job to resource manager",
    ).tag(config=True)

    req_account = Unicode(
        "",
        help="Account name string to pass to the resource manager",
    ).tag(config=True)

    req_options = Unicode(
        "",
        help="Other options to include into job submission script",
    ).tag(config=True)

    req_prologue = Unicode(
        "",
        help="Script to run before single user server starts.",
    ).tag(config=True)

    req_epilogue = Unicode(
        "",
        help="Script to run after single user server ends.",
    ).tag(config=True)

    req_username = Unicode()

    @default("req_username")
    def _req_username_default(self):
        return self.user.name

    # Useful IF getpwnam on submit host returns correct info for exec host
    req_homedir = Unicode()

    @default("req_homedir")
    def _req_homedir_default(self):
        return pwd.getpwnam(self.user.name).pw_dir

    req_keepvars = Unicode()

    @default("req_keepvars")
    def _req_keepvars_default(self):
        return ",".join(self.get_env().keys())

    req_keepvars_extra = Unicode(
        help="Extra environment variables which should be configured, "
        "added to the defaults in keepvars, "
        "comma separated list.",
    )

    batch_script = Unicode(
        "",
        help="Template for job submission script. Traits on this class named like req_xyz "
        "will be substituted in the template for {xyz} using string.Formatter. "
        "Must include {cmd} which will be replaced with the jupyterhub-singleuser command line.",
    ).tag(config=True)

    batchspawner_singleuser_cmd = Unicode(
        "",
        help="A wrapper which is capable of special batchspawner setup: currently sets the port on "
        "the remote host.  Not needed to be set under normal circumstances, unless path needs "
        "specification.",
    ).tag(config=True)

    # Raw output of job submission command unless overridden
    job_id = Unicode()

    # Will get the raw output of the job status command unless overridden
    job_status = Unicode()

    hub_connect_hostname = Unicode(
        "",
        help="Hostname of hub",
    ).tag(config=True)

    hub_connect_protocol = Unicode(
        "https",
        help="Protocol to connect to hub",
    ).tag(config=True)


    # Prepare substitution variables for templates using req_xyz traits
    def get_req_subvars(self):
        reqlist = [t for t in self.trait_names() if t.startswith("req_")]
        subvars = {}
        for t in reqlist:
            subvars[t[4:]] = getattr(self, t)
        if subvars.get("keepvars_extra"):
            subvars["keepvars"] += "," + subvars["keepvars_extra"]
        return subvars

    batch_submit_cmd = Unicode(
        "",
        help="Command to run to submit batch scripts. Formatted using req_xyz traits as {xyz}.",
    ).tag(config=True)


    def parse_job_id(self, output):
        "Parse output of submit command to get job id."
        return output

    def cmd_formatted_for_batch(self):
        """The command which is substituted inside of the batch script"""
        return " ".join([self.batchspawner_singleuser_cmd] + self.cmd + self.get_args())

    async def run_command(self, cmd, input=None, env=None):
        env_string = ''
        # TODO: add env variable copying
        with tempkey(self.ssh_private_key, self.ssh_cert, self.user.name) as keypath:
            ssh_cmd = ' '.join([
                'ssh',
                f'-i {str(keypath.absolute())}',
                '-o LogLevel=error',
                '-o StrictHostKeyChecking=no',
                '-o PasswordAuthentication=no',
                f'{self.user.name}@{self.req_host}',
                f'" {env_string} {cmd} "'
            ])
            proc = await asyncio.create_subprocess_shell(
                ssh_cmd,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            inbytes = None

            if input:
                inbytes = input.encode()

            try:
                out, eout = await proc.communicate(input=inbytes)
            except:
                self.log.debug("Exception raised when trying to run command: %s" % cmd)
                proc.kill()
                self.log.debug("Running command failed, killed process.")
                try:
                    out, eout = await asyncio.wait_for(proc.communicate(), timeout=2)
                    out = out.decode().strip()
                    eout = eout.decode().strip()
                    self.log.error("Subprocess returned exitcode %s" % proc.returncode)
                    self.log.error("Stdout:")
                    self.log.error(out)
                    self.log.error("Stderr:")
                    self.log.error(eout)
                    raise RuntimeError(
                        "{} exit status {}: {}".format(cmd, proc.returncode, eout)
                    )
                except asyncio.TimeoutError:
                    self.log.error(
                        "Encountered timeout trying to clean up command,"
                        " process probably killed already: %s"
                        % cmd
                    )
                    return ""
                except:
                    self.log.error(
                        "Encountered exception trying to clean up command: %s" % cmd
                    )
                    raise
            else:
                eout = eout.decode().strip()
                err = proc.returncode
                if err != 0:
                    self.log.error("Subprocess returned exitcode %s" % err)
                    self.log.error(eout)
                    raise RuntimeError(eout)

            out = out.decode().strip()
            return out

    async def _get_batch_script(self, **subvars):
        """Format batch script from vars"""
        # Could be overridden by subclasses, but mainly useful for testing
        return format_template(self.batch_script, **subvars)

    async def submit_batch_script(self):
        subvars = self.get_req_subvars()
        subvars['env'] = self._prepare_env_string()
        subvars['localport'] = self.port
        subvars['sish_private_key'] = self.sish_private_bytes.decode()
        subvars['hub_hostname'] = self.hub_connect_hostname

        # `cmd` is submitted to the batch system
        cmd = " ".join(
            (
                format_template(self.exec_prefix, **subvars),
                format_template(self.batch_submit_cmd, **subvars),
            )
        )
        # `subvars['cmd']` is what is run _inside_ the batch script,
        # put into the template.
        subvars["cmd"] = self.cmd_formatted_for_batch()
        if hasattr(self, "user_options"):
            subvars.update(self.user_options)
        script = await self._get_batch_script(**subvars)
        self.log.info("Spawner submitting job using " + cmd)
        self.log.info("Spawner submitted script:\n" + script)
        out = await self.run_command(cmd, input=script)
        try:
            self.log.info("Job submitted. cmd: " + cmd + " output: " + out)
            self.job_id = self.parse_job_id(out)
        except:
            self.log.error("Job submission failed with exit code " + out)
            self.job_id = ""
        return self.job_id

    def _prepare_env_string(self):
        env = self.get_env()

        if self.hub_connect_hostname:
            base_url = self.hub_connect_protocol + '://' + self.hub_connect_hostname
        else:
            base_url = self.ip
        
        username = self.user.name
        env['JUPYTERHUB_API_URL'] = '/'.join([base_url, '/hub/api'])
        env['JUPYTERHUB_ACTIVITY_URL'] = '/'.join([base_url, '/hub/api/users/' + username + '/activity'])

        envs = []
        for key in env:
            if key.startswith('JUPYTER') or key.startswith('JPY') or key == 'ACCESS_TOKEN':
                value = f"export {key}='{env[key]}'"
                envs.append(value)
        return '\n'.join(envs)


    # Override if your batch system needs something more elaborate to query the job status
    batch_query_cmd = Unicode(
        "",
        help="Command to run to query job status. Formatted using req_xyz traits as {xyz} "
        "and self.job_id as {job_id}.",
    ).tag(config=True)

    async def query_job_status(self):
        """Check job status, return JobStatus object."""
        if self.job_id is None or len(self.job_id) == 0:
            self.job_status = ""
            return JobStatus.NOTFOUND
        subvars = self.get_req_subvars()
        subvars["job_id"] = self.job_id
        cmd = " ".join(
            (
                format_template(self.exec_prefix, **subvars),
                format_template(self.batch_query_cmd, **subvars),
            )
        )
        self.log.debug("Spawner querying job: " + cmd)
        try:
            self.job_status = await self.run_command(cmd)
        except RuntimeError as e:
            # e.args[0] is stderr from the process
            self.job_status = e.args[0]
        except Exception as e:
            self.log.error("Error querying job " + self.job_id)
            self.job_status = ""

        if self.state_isrunning():
            return JobStatus.RUNNING
        elif self.state_ispending():
            return JobStatus.PENDING
        elif self.state_isunknown():
            return JobStatus.UNKNOWN
        else:
            return JobStatus.NOTFOUND

    batch_cancel_cmd = Unicode(
        "",
        help="Command to stop/cancel a previously submitted job. Formatted like batch_query_cmd.",
    ).tag(config=True)

    async def cancel_batch_job(self):
        subvars = self.get_req_subvars()
        subvars["job_id"] = self.job_id
        cmd = " ".join(
            (
                format_template(self.exec_prefix, **subvars),
                format_template(self.batch_cancel_cmd, **subvars),
            )
        )
        self.log.info("Cancelling job " + self.job_id + ": " + cmd)
        await self.run_command(cmd)

    def load_state(self, state):
        """load job_id from state"""
        super(BatchSpawnerBase, self).load_state(state)
        self.job_id = state.get("job_id", "")
        self.job_status = state.get("job_status", "")

    def get_state(self):
        """add job_id to state"""
        state = super(BatchSpawnerBase, self).get_state()
        if self.job_id:
            state["job_id"] = self.job_id
        if self.job_status:
            state["job_status"] = self.job_status
        return state

    def clear_state(self):
        """clear job_id state"""
        super(BatchSpawnerBase, self).clear_state()
        self.job_id = ""
        self.job_status = ""

    def make_preexec_fn(self, name):
        """make preexec fn to change uid (if running as root) before job submission"""
        return set_user_setuid(name)

    def state_ispending(self):
        "Return boolean indicating if job is still waiting to run, likely by parsing self.job_status"
        raise NotImplementedError("Subclass must provide implementation")

    def state_isrunning(self):
        "Return boolean indicating if job is running, likely by parsing self.job_status"
        raise NotImplementedError("Subclass must provide implementation")

    def state_isunknown(self):
        "Return boolean indicating if job state retrieval failed because of the resource manager"
        return None

    def state_gethost(self):
        "Return string, hostname or addr of running job, likely by parsing self.job_status"
        raise NotImplementedError("Subclass must provide implementation")

    async def poll(self):
        """Poll the process"""
        status = await self.query_job_status()
        if status in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.UNKNOWN):
            return None
        else:
            self.clear_state()
            return 1

    startup_poll_interval = Float(
        10,
        help="Polling interval (seconds) to check job state during startup",
    ).tag(config=True)

    async def start(self):
        """Start the process"""
        self.ip = self.traits()["ip"].default_value
        self.port = random_port()
        if self.server:
            self.server.port = self.port

        await self.submit_batch_script()

        # We are called with a timeout, and if the timeout expires this function will
        # be interrupted at the next yield, and self.stop() will be called.
        # So this function should not return unless successful, and if unsuccessful
        # should either raise and Exception or loop forever.
        if len(self.job_id) == 0:
            raise RuntimeError(
                "Jupyter batch job submission failure (no jobid in output)"
            )
        while True:
            status = await self.query_job_status()
            if status == JobStatus.RUNNING:
                break
            elif status == JobStatus.PENDING:
                self.log.debug("Job " + self.job_id + " still pending")
            elif status == JobStatus.UNKNOWN:
                self.log.debug("Job " + self.job_id + " still unknown")
            else:
                self.log.warning(
                    "Job "
                    + self.job_id
                    + " neither pending nor running.\n"
                    + self.job_status
                )
                self.clear_state()
                raise RuntimeError(
                    "The Jupyter batch job has disappeared"
                    " while pending in the queue or died immediately"
                    " after starting."
                )
            await gen.sleep(self.startup_poll_interval)


        self.ip = 'localhost'
        self.log.debug(f'ip={self.ip}, port={self.port}')

        self.db.commit()
        self.log.info(
            "Notebook server job {0} started at {1}:{2}".format(
                self.job_id, self.ip, self.port
            )
        )

        return self.ip, self.port

    async def stop(self, now=False):
        """Stop the singleuser server job.

        Returns immediately after sending job cancellation command if now=True, otherwise
        tries to confirm that job is no longer running."""

        self.log.info("Stopping server job " + self.job_id)
        await self.cancel_batch_job()
        if now:
            return
        for i in range(10):
            status = await self.query_job_status()
            if status not in (JobStatus.RUNNING, JobStatus.UNKNOWN):
                return
            await gen.sleep(1.0)
        if self.job_id:
            self.log.warning(
                "Notebook server job {0} at {1}:{2} possibly failed to terminate".format(
                    self.job_id, self.ip, self.port
                )
            )

    @async_generator
    async def progress(self):
        while True:
            if self.state_ispending():
                await yield_(
                    {
                        "message": "Pending in queue...",
                    }
                )
            elif self.state_isrunning():
                await yield_(
                    {
                        "message": "Cluster job running... waiting to connect",
                    }
                )
                return
            else:
                await yield_(
                    {
                        "message": "Unknown status...",
                    }
                )
            await gen.sleep(1)


class BatchSpawnerRegexStates(BatchSpawnerBase):
    """Subclass of BatchSpawnerBase that uses config-supplied regular expressions
    to interact with batch submission system state. Provides implementations of
        state_ispending
        state_isrunning
        state_gethost

    In their place, the user should supply the following configuration:
        state_pending_re - regex that matches job_status if job is waiting to run
        state_running_re - regex that matches job_status if job is running
        state_exechost_re - regex with at least one capture group that extracts
                            execution host from job_status
        state_exechost_exp - if empty, notebook IP will be set to the contents of the
            first capture group. If this variable is set, the match object
            will be expanded using this string to obtain the notebook IP.
            See Python docs: re.match.expand
    """

    state_pending_re = Unicode(
        "",
        help="Regex that matches job_status if job is waiting to run",
    ).tag(config=True)
    state_running_re = Unicode(
        "",
        help="Regex that matches job_status if job is running",
    ).tag(config=True)
    state_exechost_re = Unicode(
        "",
        help="Regex with at least one capture group that extracts "
        "the execution host from job_status output",
    ).tag(config=True)
    state_exechost_exp = Unicode(
        "",
        help="""If empty, notebook IP will be set to the contents of the first capture group.

        If this variable is set, the match object will be expanded using this string
        to obtain the notebook IP.
        See Python docs: re.match.expand""",
    ).tag(config=True)
    state_unknown_re = Unicode(
        "",
        help="Regex that matches job_status if the resource manager is not answering."
        "Blank indicates not used.",
    ).tag(config=True)

    def state_ispending(self):
        assert self.state_pending_re, "Misconfigured: define state_running_re"
        return self.job_status and re.search(self.state_pending_re, self.job_status)

    def state_isrunning(self):
        assert self.state_running_re, "Misconfigured: define state_running_re"
        return self.job_status and re.search(self.state_running_re, self.job_status)

    def state_isunknown(self):
        # Blank means "not set" and this function always returns None.
        if self.state_unknown_re:
            return self.job_status and re.search(self.state_unknown_re, self.job_status)

    def state_gethost(self):
        assert self.state_exechost_re, "Misconfigured: define state_exechost_re"
        match = re.search(self.state_exechost_re, self.job_status)
        if not match:
            self.log.error(
                "Spawner unable to match host addr in job status: " + self.job_status
            )
            return
        if not self.state_exechost_exp:
            return match.groups()[0]
        else:
            return match.expand(self.state_exechost_exp)


class UserEnvMixin:
    """Mixin class that computes values for USER, SHELL and HOME in the environment passed to
    the job submission subprocess in case the batch system needs these for the batch script."""

    def user_env(self, env):
        """get user environment"""
        env["USER"] = self.user.name
        home = False
        shell = False
        try:
            home = pwd.getpwnam(self.user.name).pw_dir
            shell = pwd.getpwnam(self.user.name).pw_shell
        except KeyError:
            pass
        if home:
            env["HOME"] = home
        if shell:
            env["SHELL"] = shell
        return env

    def get_env(self):
        """Get user environment variables to be passed to the user's job

        Everything here should be passed to the user's job as
        environment.  Caution: If these variables are used for
        authentication to the batch system commands as an admin, be
        aware that the user will receive access to these as well.
        """
        env = super().get_env()
        env = self.user_env(env)
        return env


class SlurmSpawner(UserEnvMixin, BatchSpawnerRegexStates):
    batch_script = Unicode(
        """#!/bin/bash
#SBATCH --output=.jupyterhub/logs/jupyterhub_slurmspawner_%j.log
#SBATCH --job-name=spawner-jupyterhub
#SBATCH --get-user-env=L
{% if partition  %}#SBATCH --partition={{partition}}
{% endif %}{% if runtime    %}#SBATCH --time={{runtime}}
{% endif %}{% if memory     %}#SBATCH --mem={{memory}}
{% endif %}{% if gres       %}#SBATCH --gres={{gres}}
{% endif %}{% if nprocs     %}#SBATCH --cpus-per-task={{nprocs}}
{% endif %}{% if reservation%}#SBATCH --reservation={{reservation}}
{% endif %}{% if options    %}#SBATCH {{options}}{% endif %}
set -x
set -euo pipefail

trap 'echo SIGTERM received' TERM

localport={{localport}}
ipnport=$(shuf -i8000-9999 -n1)
sish_key_path=~/.jupyterhub/sish_key
sish_outfile=~/.jupyterhub/jupyterhub_sish_client_logs
hub_hostname={{hub_hostname}}
(
    rm -f $sish_key_path
    cat << EOF > $sish_key_path
{{sish_private_key}}
EOF
    chmod 400 $sish_key_path
    echo "Local=$localport" > $sish_outfile;
    echo "Server=$ipnport" >> $sish_outfile;
    ssh -p 8080 -i $sish_key_path -T \
                -o LogLevel=error \
                -o StrictHostKeyChecking=no \
                -o UserKnownHostsFile=/dev/null \
            -R $localport:localhost:$ipnport $hub_hostname >> $sish_outfile 2>&1
) &

{{env}}
{{prologue}}
{% if srun %}{{srun}} {% endif %}{{cmd}}
echo "jupyterhub-singleuser ended gracefully"
{{epilogue}}
"""
    ).tag(config=True)

    # all these req_foo traits will be available as substvars for templated strings
    req_cluster = Unicode(
        "",
        help="Cluster name to submit job to resource manager",
    ).tag(config=True)

    req_qos = Unicode(
        "",
        help="QoS name to submit job to resource manager",
    ).tag(config=True)

    req_srun = Unicode(
        "srun",
        help="Set req_srun='' to disable running in job step, and note that "
        "this affects environment handling.  This is effectively a "
        "prefix for the singleuser command.",
    ).tag(config=True)

    req_reservation = Unicode(
        "",
        help="Reservation name to submit to resource manager",
    ).tag(config=True)

    req_gres = Unicode(
        "",
        help="Additional resources (e.g. GPUs) requested",
    ).tag(config=True)

    # outputs line like "Submitted batch job 209"
    batch_submit_cmd = Unicode("sbatch --parsable").tag(config=True)
    # outputs status and exec node like "RUNNING hostname"
    batch_query_cmd = Unicode("squeue -h -j {job_id} -o '%T %B'").tag(config=True)
    batch_cancel_cmd = Unicode("scancel {job_id}").tag(config=True)
    # use long-form states: PENDING,  CONFIGURING = pending
    #  RUNNING,  COMPLETING = running
    state_pending_re = Unicode(r"^(?:PENDING|CONFIGURING)").tag(config=True)
    state_running_re = Unicode(r"^(?:RUNNING|COMPLETING)").tag(config=True)
    state_unknown_re = Unicode(
        r"^slurm_load_jobs error: (?:Socket timed out on send/recv|Unable to contact slurm controller)"
    ).tag(config=True)
    state_exechost_re = Unicode(r"\s+((?:[\w_-]+\.?)+)$").tag(config=True)

    ssh_key_cert_url = Unicode(
        "",
        help="Token for cert exchanger url",
    ).tag(config=True)

    sish_sshkey_bits = Integer(4096).tag(config=True)

    sish_public_key_path = Unicode(
        "",
        help="Path to sish public key directory",
    ).tag(config=True)

    keygen_public_exponent = Integer(65537).tag(config=True)


    def parse_job_id(self, output):
        # make sure jobid is really a number
        try:
            # use only last line to circumvent slurm bug
            output = output.splitlines()[-1]
            id = output.split(";")[0]
            int(id)
        except Exception as e:
            self.log.error("SlurmSpawner unable to parse job ID from text: " + output)
            raise e
        return id

    async def start(self):
        await self.get_key_and_cert()
        private, public = self.sshkeygen()
        self.sish_private_bytes = private
        self.save_pubkey(public)

        await self.run_command('[ -d ~/.jupyterhub/logs ] || mkdir -p ~/.jupyterhub/logs')
        self.req_runtime = self.user_options.get('time', self.req_runtime)
        self.req_partition = self.user_options.get('partition', self.req_partition)
        grant_name = self.user_options.get('grant', '')
        if grant_name:
            self.req_options = f'--account={grant_name}'

        self.log.debug(f'{self.req_runtime=}')
        self.log.debug(f'{self.req_partition=}')
        self.log.debug(f'{self.req_options=}')

        ip, port = await super().start()
        return ip, port

    async def get_key_and_cert(self):
        headers = {
                'Authorization': f"Bearer {self.environment['ACCESS_TOKEN']}"
        }

        try:
            req = HTTPRequest(self.ssh_key_cert_url, method="GET", headers=headers)
            resp = await AsyncHTTPClient().fetch(req)
            data = json.loads(resp.body.decode('utf-8'))
            self.ssh_private_key = data['private']
            self.ssh_cert = data['cert']
        except HTTPClientError as e:
            self.log.debug(e)
            self.log.debug(f"Failure during authorization.")

    def sshkeygen(self):
        key = rsa.generate_private_key(
            backend=crypto_default_backend(),
            public_exponent=self.keygen_public_exponent,
            key_size=self.sish_sshkey_bits
        )

        private_bytes = key.private_bytes(
            crypto_serialization.Encoding.PEM,
            crypto_serialization.PrivateFormat.OpenSSH,
            crypto_serialization.NoEncryption()
        )

        public_bytes = key.public_key().public_bytes(
            crypto_serialization.Encoding.OpenSSH,
            crypto_serialization.PublicFormat.OpenSSH
        )

        return private_bytes, public_bytes

    def save_pubkey(self, pubkey):
        name = self.user.name
        path = pathlib.Path(self.sish_public_key_path) / name
        dump_to_rdonly_file(path, pubkey)

    def _options_form_default(self):
        return """
        <div class="form-group">
            <label for="time">Job limit time</label>
            <input class="form-control" type="text" name="time" value="{time}">
            </input>
        </div>
        <div class="form-group">
            <label for="partition">Partition name</label>
            <input class="form-control" type="partition" name="partition" value="{partition}">
            </input>
        </div>
        <div class="form-group">
            <label for="grant">Grant name</label>
            <input class="form-control" type="grant" name="grant" value="{grant}">
            </input>
        </div>
        """.format(
            time=self.req_runtime,
            partition=self.req_partition,
            grant='',
        )

    def options_from_form(self, formdata):
        options = {}
        options['time'] = formdata.get('time', '')[0]
        options['partition'] = formdata.get('partition', '')[0]
        options['grant'] = formdata.get('grant', '')[0]

        if not self.is_time_valid(options['time']):
            raise ValueError(f"Invalid limit time: {options['time']}")

        if options['grant'] != '' and not self.is_grant_valid(options['grant']):
            raise ValueError(f"Invalid grant: {options['grant']}")

        if options['partition'] != '' and not self.is_partition_valid(options['partition']):
            raise ValueError(f"Invalid partition: {options['partition']}")
        return options

    @staticmethod
    def is_time_valid(time_str: str):
        """
        Slurm valid time formats are:
         * minutes
         * minutes:seconds
         * hours:minutes:seconds
         * days-hours
         * days-hours:minutes
         * days-hours:minutes:seconds".

        Only format is validated, not values.
        """
        p = re.compile('\d+((:\d+(:\d+)?)|(-\d+(:\d+(:\d+)?)?))?')
        return p.fullmatch(time_str) is not None

    @staticmethod
    def is_grant_valid(grant_name: str):
        p = re.compile('plg[a-zA-Z0-9\-]+')
        return p.fullmatch(grant_name) is not None

    @staticmethod
    def is_partition_valid(partition_name: str):
        p = re.compile('plg[a-zA-Z0-9\-]+')
        return p.fullmatch(partition_name) is not None



class MultiSlurmSpawner(SlurmSpawner):
    """When slurm has been compiled with --enable-multiple-slurmd, the
    administrator sets the name of the slurmd instance via the slurmd -N
    option. This node name is usually different from the hostname and may
    not be resolvable by JupyterHub. Here we enable the administrator to
    map the node names onto the real hostnames via a traitlet."""

    daemon_resolver = Dict(
        {},
        help="Map node names to hostnames",
    ).tag(config=True)

    def state_gethost(self):
        host = SlurmSpawner.state_gethost(self)
        return self.daemon_resolver.get(host, host)

# vim: set ai expandtab softtabstop=4:
