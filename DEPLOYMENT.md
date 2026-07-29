# Deploy PressLens with GitHub Pages and Supabase

PressLens is a static Next.js application. GitHub Pages serves the interface
and retrieval catalogue, while Supabase Storage serves the reviewed video and
image assets.

## Requirements

- Node.js 20 or newer
- npm 10 or newer
- A Supabase project
- A GitHub repository with Pages configured to use GitHub Actions

## Configure Supabase

Create a public Storage bucket for media and set these variables locally:

```bash
export SUPABASE_URL="https://PROJECT_REF.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="YOUR_SECRET_KEY"
export SUPABASE_BUCKET="presslens-media"
```

Upload all media referenced by `public/demo/manifest.json`:

```bash
npm run upload:supabase
```

To upload only broadcast and canonical videos:

```bash
npm run upload:supabase -- --videos-only
```

The command prints the public media endpoint required by the application.

## Configure GitHub Pages

In **Settings → Pages**, select **GitHub Actions** as the source. Add the
Supabase public media endpoint under **Settings → Secrets and variables →
Actions**:

```text
MEDIA_ASSET_BASE_URL=https://PROJECT_REF.supabase.co/storage/v1/object/public/presslens-media
```

The workflow in `.github/workflows/deploy-pages.yml` builds the application
with the repository base path and deploys the static export.

## Deploy

Push the reviewed catalogue and application changes to `main`, or run
**Deploy PressLens to GitHub Pages** manually from the Actions tab:

```bash
git push origin main
```

The published URL follows this form:

```text
https://USERNAME.github.io/REPOSITORY/
```

## Local production test

```bash
export NEXT_PUBLIC_BASE_PATH=""
export NEXT_PUBLIC_MEDIA_ASSET_BASE_URL=""
npm install
npm run build
python3 -m http.server 4173 --directory out
```

An empty media endpoint uses files under the local `public/demo` directory.
