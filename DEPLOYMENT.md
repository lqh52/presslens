# Deploy PressLens with GitHub Pages and Supabase

The deployed application is static. GitHub Pages serves the interface,
catalogue, and precomputed document vectors. Supabase Storage serves the video
and image assets. MiniLM query inference, cosine similarity, and BM25 all run
inside the visitor's browser.

## 1. Confirm video-sharing permission

The current clips originate from NDA-controlled SoccerNet access. Confirm that
the intended audience and bucket visibility comply with the applicable dataset
terms before uploading any video. The uploader below creates a public bucket
and is suitable only when that visibility is permitted.

## 2. Create a Supabase project

1. Create a free project at <https://supabase.com/dashboard>.
2. Open **Project Settings → API**.
3. Copy the project URL.
4. Create or copy a secret key (`sb_secret_...`). This credential is used only
   by the upload command and is kept out of `NEXT_PUBLIC_*` variables and
   source control. A legacy `service_role` key also works, but Supabase
   recommends secret keys for new projects.

## 3. Prepare the retained assets and browser index

From the repository root:

```bash
npm install
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-research.txt

python scripts/build_reviewed_web_demo.py
npm run prepare:deploy
```

On Windows PowerShell, activate the environment with
`.\.venv\Scripts\Activate.ps1`.

The reviewed web builder removes obsolete files only from `public/demo`.
Original videos, tracking states, graphs, and annotations under `data/` remain
untouched.

## 4. Upload media to Supabase Storage

Set the credentials in your shell:

```bash
export SUPABASE_URL="https://PROJECT_REF.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="sb_secret_YOUR_SECRET_KEY"
export SUPABASE_BUCKET="presslens-media"

node scripts/upload_supabase_assets.mjs
```

The script uploads only media referenced by the current manifest. At the end it
prints the public media endpoint as:

```text
NEXT_PUBLIC_MEDIA_ASSET_BASE_URL=<generated endpoint>
```

Keep that URL. The service-role key is not needed by the deployed website.

## 5. Create and configure the GitHub repository

1. Create a GitHub repository, for example `presslens`.
2. Push this project to its `main` branch.
3. Open **Settings → Pages** and set **Source** to **GitHub Actions**.
4. Open **Settings → Secrets and variables → Actions → Secrets**.
5. Add:

   - Name: `MEDIA_ASSET_BASE_URL`
   - Value: the endpoint printed by the upload command.

The workflow automatically uses `/<repository-name>` as the GitHub Pages base
path and stops with an explicit error when the media endpoint is absent. If the
repository itself is named `<username>.github.io`, remove the
`NEXT_PUBLIC_BASE_PATH` line from `.github/workflows/deploy-pages.yml`.

## 6. Deploy

Push to `main`, or run **Deploy PressLens to GitHub Pages** manually from the
Actions tab. The resulting URL is:

```text
https://USERNAME.github.io/REPOSITORY/
```

On the first search, the browser downloads and caches the quantized
`Xenova/all-MiniLM-L6-v2` model from Hugging Face. Later searches use the cached
model. Video media is loaded from Supabase.

## 7. Update the deployed dataset

After changing review decisions or clips:

1. Repeat steps 3 and 4.
2. Commit the updated files in `deploy/catalog/`.
3. Push to `main`.

Because uploaded objects use a one-year cache header, use new filenames when
the bytes of an existing clip change, or purge/re-upload the object before
testing.

## Local production test

```bash
export NEXT_PUBLIC_BASE_PATH=""
export NEXT_PUBLIC_MEDIA_ASSET_BASE_URL=""
npm install
npm run build
python3 -m http.server 4173 --directory out
```

The empty Supabase URL makes this test use the local `public/demo` media.
