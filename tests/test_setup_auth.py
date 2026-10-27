"""scripts/perlmutter/setup.sh: the GitHub token stays in memory (credential helper reads it from the environment) and the check
fails if it shows up in .git/config, a credential file or a remote URL. Runs bash with a FAKE token; the real GH_TOKEN, if set,
is removed from the child environment."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parents[1] / "scripts" / "perlmutter" / "setup.sh"
FAKE = "ghp_FAKEtoken0123456789abcdefFAKE"
BASH = shutil.which("bash")
GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(BASH is None or GIT is None, reason="needs bash and git")


def run(script, tmp_path, token=FAKE, **extra):
    env = {k: v for k, v in os.environ.items() if k not in ("GH_TOKEN", "GITHUB_TOKEN", "TF_GIT_URL")}
    env.update(HOME=str(tmp_path / "home"), XDG_CONFIG_HOME=str(tmp_path / "home" / ".config"), GIT_CONFIG_NOSYSTEM="1", **extra)
    if token:
        env["GH_TOKEN"] = token
    (tmp_path / "home").mkdir(exist_ok=True)
    src = SETUP.as_posix()
    # gh is shadowed by a failing function, so the GH_TOKEN path is the one tested even where gh is logged in
    return subprocess.run([BASH, "-c", f'source "{src}"; gh() {{ return 1; }}; {script}'], cwd=tmp_path, env=env,
                          capture_output=True, text=True)


def make_repo(tmp_path, url="https://github.com/nareshmeena12/tideforge.git"):
    r = tmp_path / "repo"
    subprocess.run([GIT, "init", "-q", str(r)], check=True)
    subprocess.run([GIT, "-C", str(r), "remote", "add", "origin", url], check=True)
    return r


def test_no_token_in_the_script():
    text = SETUP.read_text()
    assert "$GH_TOKEN@" not in text and "${GH_TOKEN}@" not in text
    assert "credential.helper=!" in text and "credential.helper= " in text   # helper from env, stored helpers cleared


def test_helper_hands_the_token_to_git_without_storing_it(tmp_path):
    p = run('printf "protocol=https\\nhost=github.com\\n\\n" | tf_git credential fill', tmp_path)
    assert p.returncode == 0, p.stderr
    assert f"password={FAKE}" in p.stdout and "username=x-access-token" in p.stdout
    home = tmp_path / "home"
    assert not any(FAKE in f.read_text(errors="ignore") for f in home.rglob("*") if f.is_file())


def test_check_passes_on_a_clean_clone(tmp_path):
    r = make_repo(tmp_path)
    p = run(f'tf_check_no_token "{r.as_posix()}"', tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr


def test_check_fails_if_the_token_is_in_git_config(tmp_path):
    r = make_repo(tmp_path)
    with open(r / ".git" / "config", "a") as f:
        f.write(f"[http]\n\textraheader = AUTHORIZATION: bearer {FAKE}\n")
    p = run(f'tf_check_no_token "{r.as_posix()}"', tmp_path)
    assert p.returncode == 3 and ".git/config" in p.stdout
    assert FAKE not in p.stdout + p.stderr                                      # the message never prints the token


def test_check_fails_if_the_token_is_in_a_credential_file(tmp_path):
    r = make_repo(tmp_path)
    (tmp_path / "home").mkdir(exist_ok=True)
    (tmp_path / "home" / ".git-credentials").write_text(f"https://x-access-token:{FAKE}@github.com\n")
    p = run(f'tf_check_no_token "{r.as_posix()}"', tmp_path)
    assert p.returncode == 3 and ".git-credentials" in p.stdout


def test_check_fails_on_a_credentialed_remote_url_even_without_the_token(tmp_path):
    r = make_repo(tmp_path, url="https://someone:secret@github.com/nareshmeena12/tideforge.git")
    p = run(f'tf_check_no_token "{r.as_posix()}"', tmp_path, token=None)
    assert p.returncode == 3 and "remote URL" in p.stdout


def test_no_credentials_is_an_error(tmp_path):
    p = run("tf_git ls-remote origin", tmp_path, token=None)
    assert p.returncode == 2 and "no GitHub credentials" in p.stdout
