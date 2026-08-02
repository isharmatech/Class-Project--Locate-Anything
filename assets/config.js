/* =========================================================
   Demo-day configuration — ONE-LINE EDIT TARGET
   ---------------------------------------------------------
   The live demo is now hosted on a Hugging Face Space (ZeroGPU),
   so the URL below is PERSISTENT — it no longer changes every
   time you re-run a notebook. The Space URL stays the same as
   long as the Space exists.

   If GRADIO_LIVE_URL is empty (""), the Try It Yourself section
   automatically shows the fallback message + backup image/video
   instead of a blank iframe.
   ========================================================= */

window.DEMO_CONFIG = {
  // Persistent Hugging Face Space URL (ZeroGPU-backed Gradio app).
  // This stays the same for the life of the Space — no demo-day scramble.
  GRADIO_LIVE_URL: "https://cyberfrost7-locate-anything-demo.hf.space",

  // Backup recording shown when the live link is offline (Space paused/building).
  // Drop a screen-recording export at assets/demo-walkthrough.mp4
  // (or point this at a hosted URL). Leave "" to hide the video block.
  BACKUP_VIDEO_URL: "assets/demo-walkthrough.mp4",

  // Still image shown if BACKUP_VIDEO_URL is empty or the video file
  // is missing/unreachable, so the fallback never renders broken.
  BACKUP_IMAGE_URL: "assets/images/demo/demo_cell_011_out_00.png",

  // Seconds to wait for the Gradio iframe before falling back.
  IFRAME_LOAD_TIMEOUT_MS: 30000,
};
