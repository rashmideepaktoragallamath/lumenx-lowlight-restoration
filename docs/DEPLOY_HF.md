# Deploy on Hugging Face Spaces (free CPU)
1. huggingface.co > New Space > name `lumenx-lowlight` > SDK **Gradio** > Public > CPU basic.
2. In the cloned repo (weights already trained and committed in `weights/`):
   ```bash
   git remote add space https://huggingface.co/spaces/<hf-username>/lumenx-lowlight
   git push space main --force
   ```
   Password = a HF access token with **write** permission (Settings > Access Tokens). The README front-matter is already set.
3. Wait for "Running" (first build 5 to 10 min: torch). Open the URL on a phone and a laptop, upload `samples/street_before.png`, expect a result in under about 10 s.
4. If the build fails, open the Space "Logs" tab and fix the named package. Reduce the default max side in `app.py` if it is slow.
5. Put the URL in README.md, docs/DESIGN.md and the submission form.
