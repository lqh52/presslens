#!/usr/bin/env node

import { createClient } from "@supabase/supabase-js";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { config } from "dotenv";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.join(__dirname, "..");

// Load .env file
config({ path: path.join(PROJECT_ROOT, ".env") });

const supabaseUrl = process.env.SUPABASE_URL?.replace("/rest/v1/", "");
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const bucketName = process.env.SUPABASE_BUCKET || "presslens-media";

if (!supabaseUrl || !supabaseKey) {
  console.error("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env");
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

async function downloadFile(remotePath, localPath) {
  try {
    const dir = path.dirname(localPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    console.log(`Downloading: ${remotePath}`);
    const { data, error } = await supabase.storage
      .from(bucketName)
      .download(remotePath);

    if (error) {
      console.error(`  ✗ Failed: ${error.message}`);
      return false;
    }

    const buffer = await data.arrayBuffer();
    fs.writeFileSync(localPath, Buffer.from(buffer));
    console.log(`  ✓ Saved to ${localPath}`);
    return true;
  } catch (err) {
    console.error(`  ✗ Error: ${err.message}`);
    return false;
  }
}

async function main() {
  console.log(`Supabase URL: ${supabaseUrl}`);
  console.log(`Bucket: ${bucketName}`);
  console.log(`Destination: ${PROJECT_ROOT}/public/demo\n`);

  // Read manifest to get list of files to download
  const manifestPath = path.join(PROJECT_ROOT, "public/demo/manifest.json");
  if (!fs.existsSync(manifestPath)) {
    console.error("Error: manifest.json not found at", manifestPath);
    process.exit(1);
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));

  const filesToDownload = new Set();

  // Extract video paths from videos array
  manifest.videos?.forEach((video) => {
    const filename = path.basename(video.path);
    filesToDownload.add(filename);
  });

  // Extract media paths from clips
  manifest.clips?.forEach((clip) => {
    [clip.video, clip.canonicalImage, clip.canonicalVideo, clip.thumbnail].forEach(
      (mediaPath) => {
        if (mediaPath) {
          const filename = path.basename(mediaPath);
          filesToDownload.add(filename);
        }
      },
    );
  });

  console.log(`Found ${filesToDownload.size} files to download:\n`);

  let successCount = 0;
  let failureCount = 0;

  for (const filename of Array.from(filesToDownload).sort()) {
    let remotePath;

    // Determine the correct remote path structure
    if (filename.startsWith("frames/")) {
      remotePath = filename;
    } else {
      remotePath = filename;
    }

    const localPath = path.join(PROJECT_ROOT, "public/demo", filename);

    const success = await downloadFile(remotePath, localPath);
    if (success) {
      successCount += 1;
    } else {
      failureCount += 1;
    }
  }

  console.log(`\nDownload complete: ${successCount} succeeded, ${failureCount} failed`);
  if (failureCount > 0) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
