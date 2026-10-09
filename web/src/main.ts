/**
 * App entry point. Block 5 of todo.md replaces this with the real upload/convert UI.
 * For now it reports which pipeline stages are still unimplemented, so the
 * placeholder does not pretend to work.
 */

const stages = [
  ["Block 1", "Phase 0 spike — browser viability", "done"],
  ["Block 2", "Pure-logic port to MusicXML", "not started"],
  ["Block 3", "onnxruntime-web inference layer", "not started"],
  ["Block 4", "OpenCV pipeline port", "not started"],
  ["Block 5", "UI, Verovio preview, PDF input", "not started"],
  ["Block 6", "Parity harness and CI", "not started"],
] as const;

const root = document.getElementById("app");
if (root) {
  root.innerHTML = `
    <main style="font:15px/1.6 system-ui,sans-serif;max-width:46rem;margin:3rem auto;padding:0 1rem">
      <h1 style="font-size:1.4rem">homr — client-side build</h1>
      <p>
        A browser port of <a href="https://github.com/liebharc/homr">homr</a>, an optical
        music recognition system. Everything runs on your machine: no image is uploaded.
      </p>
      <h2 style="font-size:1rem;margin-top:2rem">Progress</h2>
      <table style="border-collapse:collapse;width:100%">
        <tbody>
          ${stavesRow(stages)}
        </tbody>
      </table>
      <p style="margin-top:2rem;color:#666">
        See <code>todo.md</code> in the repository for the full plan, and
        <a href="/spike.html">the Phase 0 spike</a> for the measured results.
      </p>
    </main>`;
}

function stavesRow(rows: readonly (readonly [string, string, string])[]): string {
  return rows
    .map(
      ([block, label, state]) => `
      <tr>
        <td style="padding:.35rem .5rem;border-bottom:1px solid #eee;white-space:nowrap">${block}</td>
        <td style="padding:.35rem .5rem;border-bottom:1px solid #eee">${label}</td>
        <td style="padding:.35rem .5rem;border-bottom:1px solid #eee;text-align:right;color:${
          state === "done" ? "#157f3d" : "#a06000"
        }">${state}</td>
      </tr>`,
    )
    .join("");
}