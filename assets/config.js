/* =========================================================
   Demo-day configuration — ONE-LINE EDIT TARGET
   ---------------------------------------------------------
   The live demo is hosted on a Hugging Face Space (ZeroGPU),
   so the URL below is PERSISTENT — it does not change every
   time you re-run a notebook. The Space URL stays the same
   for the life of the Space.

   If GRADIO_LIVE_URL is empty (""), or the iframe fails to
   load because the Space is asleep, the Try It Yourself
   section automatically shows the "open the Space" fallback
   message instead of a blank iframe.
   ========================================================= */

window.DEMO_CONFIG = {
  // Persistent Hugging Face Space URL (ZeroGPU-backed Gradio app).
  // This stays the same for the life of the Space — no demo-day scramble.
  GRADIO_LIVE_URL: "https://cyberfrost7-locate-anything-demo.hf.space",

  // Seconds to wait for the Gradio iframe before falling back.
  IFRAME_LOAD_TIMEOUT_MS: 30000,
};
