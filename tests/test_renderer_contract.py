from __future__ import annotations

import json
from pathlib import Path
import subprocess
import textwrap

from orchestrator_demo.a2ui_support.approval_canvas import build_approval_canvas
from orchestrator_demo.contracts import AgentDescriptor, ExecutionPlan, PlanStep


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = REPOSITORY_ROOT / "orchestrator_demo" / "app" / "static"


def _descriptor(agent_id: str, display_name: str) -> AgentDescriptor:
    return AgentDescriptor(
        agent_id=agent_id,
        display_name=display_name,
        capabilities=[agent_id.replace("_", " ")],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[],
        execution_mode="local_llm",
    )


def _approval_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_renderer_behavior",
        objective="Prepare me for tomorrow's meeting with ABC Manufacturing.",
        detected_intents=["meeting_prep", "relationship_summary"],
        selected_agents=["relationship_summary", "synthesis"],
        steps=[
            PlanStep(
                step_id="step_relationship_summary",
                agent_id="relationship_summary",
                instruction="Summarize the relationship history.",
                expected_output="Relationship history and open follow-ups.",
                data_source_categories=["relationship_history"],
                parallel_group="meeting_context",
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Create the RM-ready meeting brief.",
                depends_on=["step_relationship_summary"],
                expected_output="Final meeting preparation brief.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
        data_source_categories=["relationship_history"],
        risk_notes=["Synthetic demo data only."],
        approval_surface_id="surface_plan_renderer_behavior",
        plan_version=3,
    )


def _renderer_behavior_response() -> dict[str, object]:
    plan = _approval_plan()
    approval_messages = build_approval_canvas(
        plan,
        agent_descriptors=[
            _descriptor("relationship_summary", "Relationship Summary Agent"),
            _descriptor("synthesis", "Synthesis Agent"),
        ],
    )
    downstream_messages = [
        {
            "version": "0.1",
            "createSurface": {
                "surfaceId": "surface_product_opportunity_1",
                "catalogId": "basic",
            },
        },
        {
            "version": "0.1",
            "updateComponents": {
                "surfaceId": "surface_product_opportunity_1",
                "components": [
                    {
                        "component": "Card",
                        "id": "root",
                        "child": "downstream_text",
                    },
                    {
                        "component": "Text",
                        "id": "downstream_text",
                        "text": (
                            "Product recommendation "
                            "<script>globalThis.__executed = true</script>"
                        ),
                    },
                ],
            },
        },
    ]
    return {
        "a2uiParts": [
            {"data": message}
            for message in [*approval_messages, *downstream_messages]
        ],
        "statusEvents": [
            {
                "status": "approval_required",
                "message": "Plan approval is pending.",
                "taskId": "task_renderer_1",
                "planId": plan.plan_id,
            },
            {
                "status": "step_completed",
                "message": "Relationship summary completed.",
                "taskId": "task_renderer_1",
                "planId": plan.plan_id,
            },
        ],
        "artifacts": {
            "final_response": {
                "agent_id": "synthesis",
                "content": "Draft meeting brief ready.",
            }
        },
    }


def test_renderer_static_shell_exposes_required_local_endpoints() -> None:
    # Arrange
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "renderer.js").read_text(encoding="utf-8")

    # Act
    visible_surfaces = [
        'id="approval-surfaces"',
        'id="status-updates"',
        'id="artifact-list"',
        'id="downstream-surfaces"',
    ]

    # Assert
    for element_id in visible_surfaces:
        assert element_id in html
    assert 'src="/static/renderer.js"' in html
    assert "fetch('/api/request'" in script
    assert "fetch('/api/user-action'" in script
    assert "fetch('/api/status'" in script
    assert "fetch('/api/artifacts'" in script


def test_renderer_behaves_as_trusted_dom_mapper_for_a2ui_loop(tmp_path: Path) -> None:
    # Arrange
    response = _renderer_behavior_response()
    node_script = tmp_path / "renderer_behavior_test.cjs"
    node_script.write_text(
        textwrap.dedent(
            f"""
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');

            class MiniNode {{
              constructor() {{
                this.parentNode = null;
              }}

              get textContent() {{
                return '';
              }}

              querySelectorAll() {{
                return [];
              }}

              querySelector() {{
                return null;
              }}
            }}

            class MiniTextNode extends MiniNode {{
              constructor(text) {{
                super();
                this.nodeType = 3;
                this.text = String(text);
              }}

              get textContent() {{
                return this.text;
              }}

              set textContent(value) {{
                this.text = String(value);
              }}
            }}

            class MiniElement extends MiniNode {{
              constructor(tagName) {{
                super();
                this.nodeType = 1;
                this.tagName = tagName.toUpperCase();
                this.children = [];
                this.dataset = {{}};
                this.eventListeners = new Map();
                this.className = '';
                this.id = '';
                this.type = '';
                this.value = '';
                this.storedText = '';
              }}

              get textContent() {{
                if (this.children.length > 0) {{
                  return this.children.map((child) => child.textContent).join('');
                }}
                return this.storedText;
              }}

              set textContent(value) {{
                this.storedText = String(value);
                this.children = [];
              }}

              set innerHTML(value) {{
                this.storedText = '';
                this.children = [];
                const raw = String(value);
                if (raw.includes('<script')) {{
                  const script = new MiniElement('script');
                  script.textContent = raw;
                  this.appendChild(script);
                  return;
                }}
                this.textContent = raw.replace(/<[^>]+>/g, '');
              }}

              appendChild(child) {{
                child.parentNode = this;
                this.children.push(child);
                return child;
              }}

              replaceChildren(...children) {{
                this.children = [];
                this.storedText = '';
                for (const child of children) {{
                  this.appendChild(child);
                }}
              }}

              remove() {{
                if (!this.parentNode) {{
                  return;
                }}
                this.parentNode.children = this.parentNode.children.filter(
                  (child) => child !== this,
                );
                this.parentNode = null;
              }}

              addEventListener(type, callback) {{
                if (!this.eventListeners.has(type)) {{
                  this.eventListeners.set(type, []);
                }}
                this.eventListeners.get(type).push(callback);
              }}

              dispatchEvent(event) {{
                for (const callback of this.eventListeners.get(event.type) || []) {{
                  callback(event);
                }}
              }}

              click() {{
                this.dispatchEvent({{ type: 'click', target: this }});
              }}

              querySelectorAll(selector) {{
                const results = [];
                const wantedTag = selector.toUpperCase();
                const visit = (node) => {{
                  if (node.nodeType !== 1) {{
                    return;
                  }}
                  if (node.tagName === wantedTag) {{
                    results.push(node);
                  }}
                  for (const child of node.children) {{
                    visit(child);
                  }}
                }};
                for (const child of this.children) {{
                  visit(child);
                }}
                return results;
              }}

              querySelector(selector) {{
                return this.querySelectorAll(selector)[0] || null;
              }}
            }}

            class MiniDocument extends MiniElement {{
              constructor() {{
                super('#document');
                this.body = new MiniElement('body');
                this.appendChild(this.body);
                this.readyListeners = [];
              }}

              createElement(tagName) {{
                return new MiniElement(tagName);
              }}

              createTextNode(text) {{
                return new MiniTextNode(text);
              }}

              addEventListener(type, callback) {{
                if (type === 'DOMContentLoaded') {{
                  this.readyListeners.push(callback);
                  return;
                }}
                super.addEventListener(type, callback);
              }}

              getElementById(id) {{
                let found = null;
                const visit = (node) => {{
                  if (found || node.nodeType !== 1) {{
                    return;
                  }}
                  if (node.id === id) {{
                    found = node;
                    return;
                  }}
                  for (const child of node.children) {{
                    visit(child);
                  }}
                }};
                visit(this);
                return found;
              }}
            }}

            function mount(document, tagName, id) {{
              const element = document.createElement(tagName);
              element.id = id;
              document.body.appendChild(element);
              return element;
            }}

            function byVisibleText(root, tagName, text) {{
              return root
                .querySelectorAll(tagName)
                .find((element) => element.textContent.includes(text));
            }}

            async function main() {{
              const response = {json.dumps(response)};
              const rendererPath = {json.dumps(str(STATIC_ROOT / "renderer.js"))};
              const document = new MiniDocument();
              mount(document, 'form', 'request-form');
              mount(document, 'textarea', 'request-input');
              mount(document, 'section', 'approval-surfaces');
              mount(document, 'ol', 'status-updates');
              mount(document, 'pre', 'artifact-list');
              mount(document, 'section', 'downstream-surfaces');

              const userActions = [];
              const fetch = async (url, options = {{}}) => {{
                if (url === '/api/user-action') {{
                  userActions.push(JSON.parse(options.body));
                  return {{
                    ok: true,
                    json: async () => ({{
                      status: 'draft_updated',
                      a2uiParts: [],
                      statusEvents: [],
                      artifacts: {{}},
                    }}),
                  }};
                }}
                if (url === '/api/status') {{
                  return {{ ok: true, json: async () => ({{ statusEvents: response.statusEvents }}) }};
                }}
                if (url === '/api/artifacts') {{
                  return {{
                    ok: true,
                    json: async () => ({{
                      artifacts: response.artifacts,
                      a2uiParts: response.a2uiParts,
                    }}),
                  }};
                }}
                throw new Error(`Unexpected fetch ${{url}}`);
              }};

              const context = {{
                document,
                fetch,
                console,
                Map,
                Error,
                JSON,
                String,
                Array,
                Object,
                Promise,
                setImmediate,
              }};
              context.window = context;
              context.globalThis = context;
              vm.createContext(context);
              const source = fs.readFileSync(rendererPath, 'utf8')
                + '\\nthis.__rendererApi = {{ handleTransportResponse }};';
              vm.runInContext(source, context, {{ filename: 'renderer.js' }});

              context.__rendererApi.handleTransportResponse(response);

              const approvalText = document.getElementById('approval-surfaces').textContent;
              const downstreamText = document.getElementById('downstream-surfaces').textContent;
              const statusText = document.getElementById('status-updates').textContent;
              const artifactText = document.getElementById('artifact-list').textContent;
              assert.match(approvalText, /Approval plan/);
              assert.match(approvalText, /Prepare me for tomorrow's meeting/);
              assert.match(approvalText, /Selected agents: relationship_summary/);
              assert.match(approvalText, /step_relationship_summary/);
              assert.match(downstreamText, /Product recommendation/);
              assert.match(downstreamText, /<script>globalThis.__executed = true<\\/script>/);
              assert.equal(document.querySelector('script'), null);
              assert.equal(context.__executed, undefined);
              assert.match(statusText, /approval_required: Plan approval is pending/);
              assert.match(statusText, /step_completed: Relationship summary completed/);
              assert.match(artifactText, /Draft meeting brief ready/);

              const approvalRoot = document.getElementById('approval-surfaces');
              const instructionInput = approvalRoot
                .querySelectorAll('textarea')
                .find((element) => (
                  element.parentNode.textContent.includes(
                    'Instruction for step_relationship_summary',
                  )
                ));
              assert.ok(instructionInput);
              instructionInput.value = 'Prioritize covenant follow-ups.';
              instructionInput.dispatchEvent({{ type: 'input', target: instructionInput }});
              const addInstruction = byVisibleText(
                approvalRoot,
                'button',
                'Add instruction to step_relationship_summary',
              );
              assert.ok(addInstruction);
              addInstruction.click();
              await new Promise((resolve) => setImmediate(resolve));

              assert.equal(userActions.length, 1);
              assert.deepEqual(userActions[0], {{
                userAction: {{
                  type: 'add_instruction',
                  surfaceId: 'surface_plan_renderer_behavior',
                  payload: {{
                    planId: 'plan_renderer_behavior',
                    planVersion: 3,
                    editedPlanVersion: 3,
                    stepId: 'step_relationship_summary',
                    instruction: 'Prioritize covenant follow-ups.',
                  }},
                }},
              }});
            }}

            main().catch((error) => {{
              console.error(error);
              process.exit(1);
            }});
            """
        ),
        encoding="utf-8",
    )

    # Act
    completed = subprocess.run(
        ["node", str(node_script)],
        check=False,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    # Assert
    assert completed.returncode == 0, completed.stderr


def test_renderer_maps_basic_catalog_payloads_to_trusted_components() -> None:
    # Arrange
    script = (STATIC_ROOT / "renderer.js").read_text(encoding="utf-8")

    # Act
    supported_components = {
        "Column",
        "Row",
        "Text",
        "Button",
        "TextField",
        "Card",
        "Table",
        "Accordion",
        "Timeline",
        "Status",
    }

    # Assert
    assert "const COMPONENT_RENDERERS" in script
    for component in supported_components:
        assert f"{component}:" in script
    assert "document.createElement" in script
    assert ".textContent" in script
    assert "renderUnsupportedComponent" in script


def test_renderer_emits_structured_user_actions_with_surface_and_plan_metadata() -> None:
    # Arrange
    script = (STATIC_ROOT / "renderer.js").read_text(encoding="utf-8")

    # Act / Assert
    assert "function buildUserAction" in script
    assert "type: context.type" in script
    assert "surfaceId: context.surfaceId" in script
    assert "payload: resolvePayload(context.payload || {})" in script
    assert "planId" in script
    assert "planVersion" in script
    assert "editedPlanVersion" in script


def test_renderer_does_not_execute_agent_provided_code_or_html() -> None:
    # Arrange
    script = (STATIC_ROOT / "renderer.js").read_text(encoding="utf-8")

    # Act
    unsafe_tokens = [
        ".innerHTML",
        ".outerHTML",
        "insertAdjacentHTML",
        "eval(",
        "new Function",
        "setAttribute('onclick'",
        'setAttribute("onclick"',
    ]

    # Assert
    for token in unsafe_tokens:
        assert token not in script
    assert ".textContent" in script
    assert "addEventListener('click'" in script
