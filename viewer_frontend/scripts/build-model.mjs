import { copyFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const frontendDir = path.resolve(scriptDir, "..");
const staticDir = path.resolve(
  frontendDir,
  "..",
  "src",
  "rag_tag",
  "viewer",
  "static"
);
const workerSource = path.join(
  frontendDir,
  "node_modules",
  "@thatopen",
  "fragments",
  "dist",
  "Worker",
  "worker.mjs"
);
const workerTargetDir = path.join(staticDir, "vendor", "thatopen");
const workerTarget = path.join(workerTargetDir, "worker.mjs");

await mkdir(workerTargetDir, { recursive: true });
await copyFile(workerSource, workerTarget);

await build({
  entryPoints: [path.join(frontendDir, "src", "model-app.js")],
  bundle: true,
  format: "esm",
  minify: true,
  outfile: path.join(staticDir, "model-viewer.js"),
});
