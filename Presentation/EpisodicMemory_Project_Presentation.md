---
marp: true
theme: default
size: 16:9
paginate: true
style: |
  section {
    box-sizing: border-box;
    padding: 42px 58px 50px;
    background: radial-gradient(ellipse at 92% 4%, #173047 0, transparent 36%), #0b1220;
    color: #edf3fa;
    font-family: Aptos, Inter, Arial, sans-serif;
    font-size: 18pt;
    line-height: 1.22;
  }
  h1, h2, h3 {
    color: #f8fbff;
    font-size: 22pt;
    line-height: 1.12;
    margin: 0 0 18px;
    font-weight: 700;
  }
  h3 { margin-bottom: 10px; }
  p { margin: 0 0 10px; }
  ul, ol { margin: 0; padding-left: 1.15em; }
  li { margin: 0 0 8px; }
  strong, b { color: #68dfd0; }
  code { font-size: 18pt; color: #a8e9e1; }
  a { color: #7fe9da; }
  section::after { color: #9aaabe; font-size: 18pt; }
  section.cover {
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.cover h1 { max-width: 10.5in; margin-bottom: 20px; }
  .eyebrow {
    color: #71dfd2;
    font-weight: 700;
    letter-spacing: .09em;
    margin-bottom: 24px;
  }
  .cover-subtitle { max-width: 9.2in; color: #c9d5e4; margin-bottom: 26px; }
  .cover-meta { color: #aebbd0; margin-top: 26px; }
  .cover-rule { width: 110px; height: 5px; background: #55d8ca; border-radius: 5px; }
  .status-banner {
    display: inline-block;
    margin-top: 32px;
    padding: 10px 16px;
    border: 1px solid #317a72;
    border-radius: 12px;
    color: #a1f1e5;
    background: #102c31;
    font-weight: 700;
  }
  .grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
  .grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
  .grid4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
  .card {
    box-sizing: border-box;
    padding: 18px;
    border: 1px solid #2b3c52;
    border-radius: 15px;
    background: linear-gradient(145deg, rgba(25, 42, 62, .94), rgba(15, 27, 43, .94));
  }
  .card p:last-child, .card ul:last-child { margin-bottom: 0; }
  .card ul { padding-left: 1.05em; }
  .card li { margin-bottom: 7px; }
  .muted { color: #bbc7d7; }
  .accent { color: #71dfd2; }
  .orange { color: #ffc17a; }
  .card-label {
    color: #94a9c1;
    font-weight: 700;
    letter-spacing: .06em;
    margin-bottom: 10px;
  }
  .badge {
    display: inline-block;
    padding: 5px 10px;
    border: 1px solid #32796f;
    border-radius: 999px;
    color: #a1f1e5;
    background: #102e32;
    font-weight: 700;
  }
  .badge.future { border-color: #59697b; color: #c6cfda; background: #202b39; }
  .badge.live { border-color: #8a6132; color: #ffd49e; background: #372919; }
  .callout {
    padding: 14px 18px;
    border-left: 5px solid #57d9ca;
    border-radius: 0 12px 12px 0;
    background: rgba(25, 57, 62, .62);
  }
  .signal-track { position: relative; height: 132px; margin: 12px 0 18px; }
  .track-line {
    position: absolute;
    left: 2%; right: 2%; top: 62px;
    height: 12px;
    border-radius: 12px;
    background: repeating-linear-gradient(90deg, #26374b 0 22px, #34485f 22px 44px);
  }
  .track-line::after {
    content: ""; position: absolute; left: 0; right: 0; top: 5px;
    border-top: 2px dashed #95a5b8;
  }
  .track-caption { position: absolute; top: 88px; left: 2%; color: #aab9cb; }
  .track-event { position: absolute; top: 7px; transform: translateX(-50%); text-align: center; white-space: nowrap; }
  .track-event i { display: block; width: 20px; height: 20px; margin: 0 auto 7px; border-radius: 50%; background: #ffb86a; box-shadow: 0 0 0 6px rgba(255, 184, 106, .15); }
  .track-event.second i { background: #71dfd2; box-shadow: 0 0 0 6px rgba(113, 223, 210, .15); }
  .flow { display: grid; grid-template-columns: 1fr 28px 1fr 28px 1fr 28px 1fr; gap: 7px; align-items: stretch; }
  .flow-step { min-height: 190px; padding: 16px; border: 1px solid #2b3c52; border-radius: 14px; background: #122035; }
  .flow-step h3 { margin: 8px 0 10px; }
  .flow-num { color: #77e0d3; font-weight: 700; }
  .flow-arrow { align-self: center; color: #75ddd0; text-align: center; font-weight: 700; }
  .branch-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 15px; }
  .branch { padding: 12px 16px; border: 1px solid #2b3c52; border-radius: 12px; background: #101c2d; }
  .branch p { margin: 0; }
  .extension-card { min-height: 270px; }
  .signal-title { color: #aebbd0; margin-bottom: 6px; }
  .artifact-row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 14px; }
  .artifact { padding: 12px 14px; border-radius: 11px; background: #111c2c; border: 1px solid #26384d; }
  .artifact p { margin: 0; }
  .progress-card { min-height: 210px; }
  .result-banner {
    margin-top: 15px;
    padding: 15px 18px;
    border: 1px solid #805c34;
    border-radius: 13px;
    color: #ffdaa8;
    background: rgba(58, 42, 25, .64);
    font-weight: 700;
  }
  .final-question { margin-top: 20px; padding: 15px 18px; border: 1px solid #397f78; border-radius: 13px; background: #10282d; }
  .final-question p { margin: 0; }
---

<!-- _class: cover -->
<div class="eyebrow">MONASH UNIVERSITY · ENGINEERING PROJECT</div>

# Episodic Memory for Robots in the Wild

<p class="cover-subtitle"><strong>Benchmarking VLM recall</strong> in long, event-sparse egocentric dashcam drives</p>

<div class="cover-rule"></div>

<div class="cover-meta">William Ngo · Vedansh Malhan</div>

<div class="status-banner">SPARSE + TEMPORAL PIPELINES COMPLETE · MODEL EVALUATION IN PROGRESS</div>

<!--
Opening: This project asks a simple question with an important deployment consequence: can a vision-language model remember what happened during a long, quiet drive, rather than just describe the frame in front of it?
Transition: First, why the usual video QA setup does not answer that question.
-->

---

# The client's hard case is long, quiet and repetitive

<div class="grid2">
  <div class="card">
    <div class="card-label">TYPICAL VIDEO QA</div>
    <h3>Many visible cues</h3>
    <p class="muted">Short clips, frequent actions and distinctive scenes make it easier to answer from local visual evidence.</p>
  </div>
  <div class="card">
    <div class="card-label">OUR TARGET SETTING</div>
    <h3>Few events across a long drive</h3>
    <p class="muted">Forward-facing vehicle video in forests, rural roads and tunnels; target durations are roughly 6 minutes to 1 hour.</p>
  </div>
</div>

<div class="signal-track">
  <div class="track-line"></div>
  <div class="track-event" style="left: 27%;"><i></i><span>rare event</span></div>
  <div class="track-event second" style="left: 74%;"><i></i><span>another moment</span></div>
  <div class="track-caption">CONTINUOUS ROAD SCENE · LONG GAPS BETWEEN MEMORABLE EVENTS</div>
</div>

<div class="callout"><strong>Client need:</strong> test whether an agent can verify what occurred and when—without guessing from familiar driving patterns or inventing plausible events.</div>

<!--
The earlier report identified a gap in benchmarks built around short, event-dense clips. Here, the low signal-to-noise ratio is intentional: most frames are repetitive road scenery, and the important evidence may be separated by many minutes.
The timeline is a schematic, not a measured event distribution. The deployment motivation is reliable long-term context for autonomous vehicles and embodied agents.
-->

---

# Turn episodic memory into four measurable capabilities

<div class="callout" style="margin-bottom: 16px;"><strong>Research question:</strong> can a VLM distinguish real from absent events, remember their order, and reason over observations spread across a long drive?</div>

<div class="grid2">
  <div class="card">
    <div class="card-label">01 · SPARSE EVENT LOCALISATION</div>
    <h3>Did the rare event happen?</h3>
    <p class="muted">Existence, noteworthy-moment and deceptive-question tests.</p>
    <span class="badge">PIPELINE IMPLEMENTED</span>
  </div>
  <div class="card">
    <div class="card-label">02 · TEMPORAL ORDER</div>
    <h3>What happened first, next or before?</h3>
    <p class="muted">Tests chronology across events separated in time.</p>
    <span class="badge">PIPELINE IMPLEMENTED</span>
  </div>
  <div class="card">
    <div class="card-label">03 · SPATIAL REASONING</div>
    <h3>Where, and how far?</h3>
    <p class="muted">Grounded location and distance reasoning with few landmarks.</p>
    <span class="badge future">PLANNED</span>
  </div>
  <div class="card">
    <div class="card-label">04 · COUNTING</div>
    <h3>How many distinct objects appeared?</h3>
    <p class="muted">Tracks objects across the drive, not just within one frame.</p>
    <span class="badge future">PLANNED</span>
  </div>
</div>

<!--
The current code connects the sparse-event and temporal-order pipelines. Spatial reasoning and counting remain part of the broader benchmark design, but they are not implemented yet. Calling this out keeps the project scope ambitious without overstating the current product.
-->

---

# Three delivered extensions make the benchmark usable

<div class="grid3">
  <div class="card extension-card">
    <div class="card-label">EXTENSION 01</div>
    <h3>Sparse-event benchmark</h3>
    <p><strong>Engineering:</strong> section-level detection, duplicate review and questions with plausible false events.</p>
    <p><strong>Client value:</strong> measures both missed events and unsupported claims.</p>
  </div>
  <div class="card extension-card">
    <div class="card-label">EXTENSION 02</div>
    <h3>Temporal-order benchmark</h3>
    <p><strong>Engineering:</strong> timestamp-grounded event timelines, human review and order-sensitive questions.</p>
    <p><strong>Client value:</strong> separates remembering the event from remembering its place in the drive.</p>
  </div>
  <div class="card extension-card">
    <div class="card-label">EXTENSION 03</div>
    <h3>Cross-model QA + grading</h3>
    <p><strong>Engineering:</strong> configurable model inputs, answer parsing and automatic exact-match scoring.</p>
    <p><strong>Client value:</strong> repeatable comparisons and traceable per-question failures.</p>
  </div>
</div>

<p class="muted" style="margin-top: 14px;">Together, the extensions move the work from a concept toward an auditable evaluation workflow.</p>

<!--
The repository does not explicitly label its formal post-IPD extension list. These three are the clearest implemented additions evidenced by the code and documentation: the two task pipelines plus the shared multi-model answering and grading path. If the project brief uses different official labels, update the card titles to match it.
-->

---

# Ground truth is built with timestamps and a human review gate

<div class="flow">
  <div class="flow-step">
    <div class="flow-num">01 · SEGMENT</div>
    <h3>Timestamped frames</h3>
    <p class="muted">Sample into model-sized sections; keep true elapsed-time ranges in a manifest.</p>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-step">
    <div class="flow-num">02 · EXTRACT</div>
    <h3>VLM observations</h3>
    <p class="muted">Task prompts ask for rare events or a section-level event timeline.</p>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-step">
    <div class="flow-num">03 · RECONCILE</div>
    <h3>Order + merge</h3>
    <p class="muted">Python sorts by frame timestamp; the LLM merges duplicates, without inventing or reordering.</p>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-step">
    <div class="flow-num">04 · VERIFY</div>
    <h3>Human review</h3>
    <p class="muted">Check each event against its frame evidence before generating questions.</p>
  </div>
</div>

<div class="branch-grid">
  <div class="branch"><p><strong>Sparse output:</strong> reviewed <code>events.json</code> → existence, deceptive and noteworthy questions.</p></div>
  <div class="branch"><p><strong>Temporal output:</strong> reviewed <code>timeline.json</code> → pairwise, sequence and before/after questions.</p></div>
</div>

<!--
Key engineering decision: models do not supply the authoritative time values. Timestamped filenames and the section manifest establish the timeline; Python provides deterministic ordering. The language model is constrained to merging/filtering, and a human verifies the event list before questions are generated.
This is the quality-control chain that makes generated questions auditable rather than treating model output as ground truth.
-->

---

# Question types probe recall, not just recognition

<div class="grid2">
  <div class="card">
    <div class="card-label">SPARSE EVENT LOCALISATION</div>
    <h3>Can the model resist a plausible guess?</h3>
    <ul>
      <li><strong>Existence:</strong> did a verified event occur?</li>
      <li><strong>Deceptive:</strong> did a plausible but absent event occur? Correct answer: no.</li>
      <li><strong>Noteworthy:</strong> choose the one verified event among four candidates.</li>
    </ul>
  </div>
  <div class="card">
    <div class="card-label">TEMPORAL ORDER</div>
    <h3>Can the model recover chronology?</h3>
    <ul>
      <li><strong>Pairwise:</strong> which of two events happened first?</li>
      <li><strong>Sequence:</strong> put three to five observed events in order.</li>
      <li><strong>Before/after:</strong> identify what immediately preceded or followed an event.</li>
    </ul>
  </div>
</div>

<div class="callout" style="margin-top: 15px;">Each generated item retains event IDs and frame evidence for review; the answering model receives the question and options, not the answer key or evidence metadata.</div>

<!--
Use the examples as task definitions, not as claims about a particular drive. The sparse question design deliberately includes absence verification. The temporal sequence type is graded in order, so a correct set in the wrong chronology is still incorrect.
-->

---

# One questionnaire, configurable model inputs, automatic grading

<div class="grid3">
  <div class="card">
    <div class="card-label">1 · BLIND THE QUESTIONS</div>
    <h3>Hide the key</h3>
    <p class="muted">Remove answers and evidence from the prompt; randomize question order.</p>
  </div>
  <div class="card">
    <div class="card-label">2 · RUN THE MODEL</div>
    <h3>Record the input</h3>
    <p class="muted">Use bounded, uniformly sampled frames or a supported native-video path; save input metadata.</p>
  </div>
  <div class="card">
    <div class="card-label">3 · SCORE THE OUTPUT</div>
    <h3>Compare answer indices</h3>
    <p class="muted">Exact-match grading; sequence questions are order-sensitive.</p>
  </div>
</div>

<div class="artifact-row">
  <div class="artifact"><p><strong>answers.json</strong><br><span class="muted">model response + input metadata</span></p></div>
  <div class="artifact"><p><strong>graded_results.json</strong><br><span class="muted">per-question correctness + accuracy</span></p></div>
  <div class="artifact"><p><strong>Compare by condition</strong><br><span class="muted">model × task × drive</span></p></div>
</div>

<!--
The runner supports configurable API/local models and frame or native-video input where the model adapter permits it. Frame-mode defaults are bounded to protect context and cost. For a fair comparison, report the input mode and frame budget alongside scores; native-video and sampled-frame runs should be treated as distinct conditions unless their coverage is matched.
-->

---

# The benchmark pipelines are ready; model scores are still running

<div class="grid3">
  <div class="card progress-card">
    <div class="card-label">BENCHMARK CREATION</div>
    <h3>Sparse + temporal stages</h3>
    <p class="muted">Extraction, review/timeline construction and question generation are implemented.</p>
    <span class="badge">COMPLETE</span>
  </div>
  <div class="card progress-card">
    <div class="card-label">EVALUATION TOOLING</div>
    <h3>Questionnaire + grader</h3>
    <p class="muted">Model input adapters, answer parsing and exact-match grading are implemented.</p>
    <span class="badge">READY</span>
  </div>
  <div class="card progress-card">
    <div class="card-label">CURRENT EXPERIMENT</div>
    <h3>Answering across models</h3>
    <p class="muted">The model question-answer runs are underway; final comparisons are pending.</p>
    <span class="badge live">IN PROGRESS</span>
  </div>
</div>

<div class="result-banner">No measured model leaderboard is shown yet. The next evidence will be per-task scores plus reviewed examples of correct answers, misses and hallucinations.</div>

<p class="muted" style="margin-top: 12px;">Spatial reasoning and counting remain future implementation work.</p>

<!--
This is the current outcome: a functioning sparse and temporal question-generation pipeline plus reusable evaluation tooling. Do not report the initial paper's projected performance ceilings as measured results. Once the current runs finish, replace this status emphasis with per-category/model plots generated from the grader outputs.
-->

---

# Make the next results decision-ready

<div class="grid2">
  <div class="card">
    <div class="card-label">LIMITATIONS & TRADE-OFFS</div>
    <h3>Interpret scores carefully</h3>
    <ul>
      <li>Model-assisted event extraction still needs human verification.</li>
      <li>Frame sampling and context limits can miss brief events.</li>
      <li>Different native-video and sampled-frame inputs can affect comparability.</li>
      <li>Spatial and counting tasks are not yet implemented.</li>
    </ul>
  </div>
  <div class="card">
    <div class="card-label">RECOMMENDED NEXT STEPS</div>
    <h3>Turn runs into evidence</h3>
    <ul>
      <li>Finish model runs; report accuracy by task, drive and input condition.</li>
      <li>Audit false positives, missed events and order errors with frame evidence.</li>
      <li>Expand scene coverage and implement spatial reasoning/counting.</li>
    </ul>
  </div>
</div>

<div class="callout" style="margin-top: 15px;">Recommended reporting: matched input budgets, per-question error examples and a clear separation between measured results and projections.</div>

<!--
The frame-budget and sampling choices are engineering trade-offs: they help keep inference tractable but can reduce temporal coverage. Human review remains necessary because errors in the event list can become errors in the benchmark answer key. The first improvement should be to finish evaluation and produce interpretable task-level comparisons before making broad claims about model capability.
-->

---

# A repeatable stress test for long-horizon memory

<div class="callout">
  <strong>What the client gets today:</strong> traceable sparse-event and temporal-order evaluations built for long, low-signal driving video—plus a reusable path to compare VLMs.
</div>

<div class="grid2" style="margin-top: 16px;">
  <div class="card">
    <div class="card-label">NEXT MILESTONE</div>
    <h3>Complete the comparison</h3>
    <p class="muted">Finish model runs → publish task-level plots and failure examples → prioritize the next benchmark extension.</p>
  </div>
  <div class="card">
    <div class="card-label">CLIENT DISCUSSION</div>
    <h3>Which failure is most costly?</h3>
    <p class="muted">Missing a rare event, inventing one, or remembering it in the wrong order?</p>
  </div>
</div>

<!--
Close with the project value, then invite the client to rank the failure modes. This creates a natural bridge from the current two implemented tasks to prioritizing spatial reasoning, counting and deployment-relevant evaluation conditions.
-->
