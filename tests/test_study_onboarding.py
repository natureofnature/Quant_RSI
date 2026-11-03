"""Guided starts reuse the queue and freeze the chosen cost assumptions."""

import json
import threading
from urllib.request import urlopen

import pytest
from test_study_queue import OFFLINE, finish_queue

from quant_rsi.research import study_onboarding
from quant_rsi.research.study_queue import study_options
from quant_rsi.research.study_trigger import refresh_evidence
from quant_rsi.research.study_workbench import Workbench, make_server


def test_setup_checks_are_local_and_never_return_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-sentinel-never-return")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(study_onboarding.shutil, "which", lambda name: None)
    monkeypatch.setattr(study_onboarding, "find_spec", lambda name: object())
    workbench = Workbench(tmp_path)
    server = make_server(workbench, "setup-test")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/setup-test/setup") as response:
            body = response.read().decode()
        assert "secret-sentinel" not in body
        result = json.loads(body)
        providers = {p["id"]: p for p in result["services"]["providers"]}
        assert providers["api"]["available"] and "unverified" in providers["api"]["detail"]
        assert not providers["deepseek"]["available"]
        assert not providers["codex"]["available"]
        assert len(result["starters"]) == 3
        assert workbench.listing() == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_offline_starter_runs_real_worker_with_custom_costs_and_saved_template(tmp_path):
    starter = next(s for s in study_onboarding.starter_studies() if s["id"] == "offline")
    assert starter["options"]["llm"] == "simulated"
    assert starter["options"]["data"] == "synthetic"
    workbench = Workbench(tmp_path)
    costs = {
        "initial_cash": "25000",
        "commission_bps": "12.5",
        "slippage_bps": "7.5",
        "threshold": 0.001,
    }
    plan = workbench.save_plan(
        {
            "name": "Offline starter with explicit costs",
            "symbols": starter["symbols"],
            "options": starter["options"] | costs,
            "interval_hours": 24,
            "enabled": False,
        }
    )
    batch = workbench.run_plan(plan["id"])
    finish_queue(workbench)
    task = workbench.task(batch["tasks"][-1]["id"])
    item = workbench.item(task)
    assert item["status"] == "completed", item.get("error")
    result = json.loads((task / "study/result.json").read_bytes())
    for name, value in costs.items():
        assert result["configuration"][name] == value
        assert result["study"]["cost_stress"]["base_policy"][name] == value
        assert Workbench(tmp_path).plans()[0]["options"][name] == value
    scenarios = result["study"]["cost_stress"]["segments"]["development"]["scenarios"]
    assert [s["commission_bps"] for s in scenarios] == ["12.5", "25", "37.5"]
    assert [s["slippage_bps"] for s in scenarios] == ["7.5", "15", "22.5"]
    assert (task / "study/report.html").is_file()


@pytest.mark.parametrize(
    "change",
    [
        {"initial_cash": "0"},
        {"initial_cash": "100.001"},
        {"initial_cash": True},
        {"commission_bps": "-1"},
        {"commission_bps": "1001"},
        {"slippage_bps": "NaN"},
        {"threshold": float("inf")},
        {"threshold": True},
    ],
)
def test_invalid_execution_inputs_fail_before_enqueue(tmp_path, change):
    workbench = Workbench(tmp_path)
    with pytest.raises(ValueError):
        workbench.batch(["KO"], OFFLINE | change)
    assert not workbench.listing()


def test_cost_change_is_new_evidence_and_old_callers_keep_defaults(tmp_path):
    workbench = Workbench(tmp_path)
    fingerprints = []
    for commission in ("5", "10"):
        options = study_options(OFFLINE | {"commission_bps": commission})
        assert options["initial_cash"] == "10000" and options["slippage_bps"] == "5"
        task = workbench._enqueue("KO", options)
        manifest = json.loads((task / "task.json").read_bytes())
        receipt, _ = refresh_evidence(task, manifest)
        fingerprints.append(receipt["fingerprint"])
    assert fingerprints[0] != fingerprints[1]


def test_full_workbench_script_applies_starter_without_running_or_scheduling(tmp_path):
    import re
    import shutil
    import subprocess
    from importlib.resources import files

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the optional workbench script check")
    html = files("quant_rsi.research").joinpath("study_workbench.html").read_text()
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    payload = {
        "ids": re.findall(r'id="([^"]+)"', html),
        "names": re.findall(r'name="([^"]+)"', html),
        "defaults": study_options({}),
        "setup": {
            "services": study_onboarding.local_services(),
            "starters": study_onboarding.starter_studies(),
        },
    }
    harness = (
        "const fixture="
        + json.dumps(payload)
        + ";\n"
        + r"""
const assert = require('node:assert/strict');
class Element {
  constructor(tag='div'){this.tag=tag;this.children=[];this.hidden=false;this.value='';}
  append(...items){this.children.push(...items);}
  prepend(...items){this.children.unshift(...items);}
  replaceChildren(...items){this.children=[...items];}
  addEventListener(){}
  scrollIntoView(){}
}
const nodes=Object.fromEntries(fixture.ids.map(id=>[id,new Element()]));
const fields=Object.fromEntries(fixture.names.map(name=>{const value=fixture.defaults[name]??'';
  return [name,{value:String(value),defaultValue:String(value),checked:value===true,
    type:typeof value==='boolean'?'checkbox':'text'}];}));
fields.symbols.value='KO';fields.enabled.checked=true;
nodes['start-form'].elements=fields;
global.document={getElementById:id=>nodes[id],createElement:tag=>new Element(tag)};
global.FormData=class {constructor(form){this.fields=form.elements;}
  get(name){const f=this.fields[name];return !f||f.disabled?null:f.value;}
  has(name){const f=this.fields[name];
    return Boolean(f&&!f.disabled&&(f.type!=='checkbox'||f.checked));}};
global.setInterval=()=>0;
const requests=[];
global.fetch=async(path,opts)=>{requests.push({path,...opts});let data={};
  if(path==='setup')data=fixture.setup;
  if(path==='tasks')data={tasks:[]};
  if(path==='plans')data={plans:[]};
  if(path==='overview')data={tasks:[],histories:{}};
  return {ok:true,json:async()=>data};};
"""
    )
    assertions = r"""
(async()=>{
  await new Promise(setImmediate);
  assert.equal(nodes.starters.children.length,3);
  assert.equal(requests.filter(r=>r.method==='POST').length,0);
  nodes.starters.children[0].children.find(n=>n.tag==='button').onclick();
  assert.equal(fields.research_mode.value,'incremental');
  assert.equal(fields.llm.value,'simulated');
  assert.equal(fields.data.value,'synthetic');
  assert.equal(fields.enabled.checked,false);
  assert.equal(fields.commission_bps.value,'5');
  assert.match(nodes['run-preview'].textContent,/3 research \+ 0 diagnostic/);
  assert.equal(requests.filter(r=>r.method==='POST').length,0);
  fields.initial_cash.value='25000';fields.commission_bps.value='12.5';
  await nodes['start-form'].onsubmit({preventDefault(){}});
  const posts=requests.filter(r=>r.method==='POST');
  assert.equal(posts.length,1);assert.equal(posts[0].path,'batches');
  const submitted=JSON.parse(posts[0].body);
  assert.equal(submitted.options.initial_cash,'25000');
  assert.equal(submitted.options.commission_bps,'12.5');
  assert.equal(submitted.options.llm,'simulated');
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(
        [node, "-"], input=harness + script + assertions, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
