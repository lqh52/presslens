#!/usr/bin/env node

import { copyFile, mkdir } from "node:fs/promises";
import { join } from "node:path";

const mode = process.argv[2];
if (!["restore", "publish"].includes(mode)) {
  throw new Error("Usage: node scripts/sync_deploy_catalog.mjs restore|publish");
}

const source = mode === "restore" ? "deploy/catalog" : "public/demo";
const destination = mode === "restore" ? "public/demo" : "deploy/catalog";
await mkdir(destination, { recursive: true });

for (const name of ["manifest.json", "search-index.json"]) {
  await copyFile(join(source, name), join(destination, name));
}

console.log(`${mode === "restore" ? "Restored" : "Published"} deployment catalogue`);
