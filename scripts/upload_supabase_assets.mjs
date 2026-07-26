#!/usr/bin/env node

import { createClient } from "@supabase/supabase-js";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";

const configuredProjectUrl = process.env.SUPABASE_URL;
const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const bucket = process.env.SUPABASE_BUCKET ?? "presslens-media";
const assetRoot = process.env.PRESSLENS_ASSET_ROOT ?? "public/demo";
const videosOnly = process.argv.includes("--videos-only");

if (!configuredProjectUrl || !serviceRoleKey) {
  throw new Error(
    "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY before uploading.",
  );
}
const projectUrl = new URL(configuredProjectUrl).origin;

const contentTypes = {
  ".jpg": "image/jpeg",
  ".png": "image/png",
  ".mp4": "video/mp4",
};

const manifest = JSON.parse(
  await readFile(join(assetRoot, "manifest.json"), "utf8"),
);
const relativePaths = new Set();
for (const clip of manifest.clips) {
  for (const field of [
    "video",
    "canonicalImage",
    "canonicalVideo",
    "thumbnail",
  ]) {
    relativePaths.add(clip[field].replace(/^\/demo\//, ""));
  }
}

const supabase = createClient(projectUrl, serviceRoleKey, {
  auth: { persistSession: false, autoRefreshToken: false },
});

const { error: bucketError } = await supabase.storage.createBucket(bucket, {
  public: true,
  allowedMimeTypes: Object.values(contentTypes),
  fileSizeLimit: 50 * 1024 * 1024,
});
if (bucketError && !bucketError.message.toLowerCase().includes("already exists")) {
  throw bucketError;
}

for (const relativePath of [...relativePaths].sort()) {
  const extension = extname(relativePath).toLowerCase();
  if (videosOnly && extension !== ".mp4") continue;
  const body = await readFile(join(assetRoot, relativePath));
  const { error } = await supabase.storage
    .from(bucket)
    .upload(relativePath, body, {
      contentType: contentTypes[extension] ?? "application/octet-stream",
      cacheControl: "31536000",
      upsert: true,
    });
  if (error) throw new Error(`${relativePath}: ${error.message}`);
  process.stdout.write(`Uploaded ${relativePath}\n`);
}

console.log(
  `\nNEXT_PUBLIC_MEDIA_ASSET_BASE_URL=${projectUrl}/storage/v1/object/public/${bucket}`,
);
