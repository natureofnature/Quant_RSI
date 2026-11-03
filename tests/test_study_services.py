"""Provider setup is private and its explicit test cannot start research."""

import json
import os
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_study_onboarding import run_workbench_script
from test_study_queue import finish_queue

from quant_rsi.research import study_services
from quant_rsi.research.study_workbench import Workbench, make_server


def test_private_settings_override_only_worker_environment_and_clear_falls_back(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "server-secret")
    settings = {"provider": "api", "model": "test-model", "key": "workspace-secret"}
    metadata = study_services.save_service_settings(tmp_path, settings)
    path = tmp_path / ".provider-settings.json"
    assert path.stat().st_mode & 0o777 == 0o600
    assert "secret" not in json.dumps(metadata)
    assert metadata[1]["key_source"] == "workspace"
    assert study_services.effective_environment(tmp_path)["OPENAI_API_KEY"] == "workspace-secret"
    assert os.environ["OPENAI_API_KEY"] == "server-secret"
    study_services.save_service_settings(tmp_path, settings | {"key": ""})
    assert study_services.effective_environment(tmp_path)["OPENAI_API_KEY"] == "workspace-secret"
    metadata = study_services.save_service_settings(
        tmp_path, settings | {"key": "", "clear_key": True}
    )
    assert metadata[1]["key_source"] == "server environment"
    assert study_services.effective_environment(tmp_path)["OPENAI_API_KEY"] == "server-secret"
    assert "workspace-secret" not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("response", ['{"ok": true}', '{"ok": false}', '{"ok": 1}', "not-json"])
def test_probe_uses_existing_transport_and_retains_only_whitelisted_usage(
    tmp_path, monkeypatch, response
):
    calls = []

    def structured(client, prompt, schema):
        calls.append(client)
        client.on_usage({"input_tokens": 10, "output_tokens": 2, "private": "secret"})
        assert schema["additionalProperties"] is False
        return response

    monkeypatch.setattr(study_services.LLMHypothesizer, "structured", structured)
    code = study_services.check_connection(tmp_path, {"llm": "codex", "llm_model": ""})
    assert code == (0 if response == '{"ok": true}' else 1)
    result = json.loads((tmp_path / "connection.json").read_bytes())
    assert result["usage"] == [{"input_tokens": 10, "output_tokens": 2}]
    assert len(calls) == 1 and calls[0].timeout == 60
    assert "secret" not in json.dumps(result) and "not-json" not in json.dumps(result)


def test_failed_probe_redacts_raw_error_and_never_retries(tmp_path, monkeypatch):
    calls = []

    def fail(*args):
        calls.append(1)
        raise RuntimeError("Bearer secret-account-token")

    monkeypatch.setattr(study_services.LLMHypothesizer, "structured", fail)
    assert study_services.check_connection(tmp_path, {"llm": "api", "llm_model": "test"}) == 1
    text = (tmp_path / "connection.json").read_text()
    assert "secret-account-token" not in text and "failed" in text
    assert len(calls) == 1


def test_real_queue_injects_workspace_key_without_persisting_it_and_continues(
    tmp_path, monkeypatch
):
    binary = tmp_path / "bin"
    binary.mkdir()
    codex = binary / "codex"
    # Exercise the process boundary with an offline CLI double, not a real provider.
    codex.write_text(
        "#!/usr/bin/env python3\nimport sys,os\nfrom pathlib import Path\n"
        "assert os.environ['OPENAI_API_KEY']=='workspace-secret'\n"
        "if '--model' in sys.argv and sys.argv[sys.argv.index('--model')+1]=='fail':\n"
        " print('secret-account-token',file=sys.stderr)\n sys.exit(1)\n"
        "Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text('{\"ok\":true}')\n"
    )
    codex.chmod(0o700)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    workbench = Workbench(tmp_path / "workspace")
    study_services.save_service_settings(
        workbench.root, {"provider": "api", "model": "test", "key": "workspace-secret"}
    )
    failed = workbench.check_provider({"provider": "codex", "model": "fail"})
    good = workbench.check_provider({"provider": "codex", "model": ""})
    finish_queue(workbench)
    assert workbench.item(workbench.task(failed["id"]))["status"] == "failed"
    assert workbench.item(workbench.task(good["id"]))["status"] == "completed"
    for path in workbench.tasks.rglob("*"):
        if path.is_file():
            assert b"workspace-secret" not in path.read_bytes()
            assert b"secret-account-token" not in path.read_bytes()
    assert not list(workbench.tasks.glob("*/study"))


def test_http_settings_use_existing_origin_gate_and_do_not_expose_key(tmp_path):
    workbench = Workbench(tmp_path)
    server = make_server(workbench, "test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    base = origin + "/test-token/"
    data = json.dumps({"provider": "api", "model": "test", "key": "private-secret"}).encode()
    try:
        for headers in (
            {"Content-Type": "application/json"},
            {"Content-Type": "application/json", "Origin": origin},
        ):
            request = Request(base + "service-settings", data=data, headers=headers)
            if "Origin" not in headers:
                with pytest.raises(HTTPError) as error:
                    urlopen(request)
                assert error.value.code == 400
            else:
                with urlopen(request) as response:
                    assert b"private-secret" not in response.read()
        with urlopen(base + "setup") as response:
            body = response.read()
            assert b"private-secret" not in body and b"workspace" in body
        with pytest.raises(HTTPError) as error:
            urlopen(base + ".provider-settings.json")
        assert error.value.code == 404
        assert workbench.listing() == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_browser_service_actions_do_not_enqueue_studies_or_copy_keys():
    run_workbench_script(
        r"""
(async()=>{
  await new Promise(setImmediate);
  nodes['service-provider'].value='api';renderServiceSettings();
  nodes['service-key'].value='browser-secret';nodes['service-model'].value='changed-model';
  await nodes['save-service'].onclick();
  assert.equal(nodes['service-key'].value,'');
  assert.equal(requests.filter(r=>r.path==='service-settings').length,1);
  nodes['apply-service'].onclick();
  assert.equal(fields.llm.value,'api');assert.equal(fields.llm_model.value,'test-model');
  await nodes['test-service'].onclick();
  const probe=requests.find(r=>r.path==='provider-checks');
  assert.deepEqual(JSON.parse(probe.body),{provider:'api',model:'test-model'});
  assert.equal(requests.filter(r=>['tasks','batches','checks','plans'].includes(r.path)&&r.method==='POST').length,0);
})().catch(error=>{console.error(error);process.exitCode=1;});
""",
        extra={
            "setup": {
                "services": {"providers": [], "models": []},
                "starters": [],
                "settings": [{"provider": "api", "model": "test-model", "key_source": "workspace"}],
            }
        },
    )
