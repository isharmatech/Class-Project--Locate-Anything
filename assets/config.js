/* =========================================================
   Demo-day configuration — ONE-LINE EDIT TARGET
   ---------------------------------------------------------
   Before presenting, open colab_demo.ipynb in Colab, run all
   cells top to bottom, copy the https://xxxxx.gradio.live URL
   the last cell prints, and paste it below into GRADIO_LIVE_URL.
   Then commit + push so GitHub Pages rebuilds with the live link.

   If GRADIO_LIVE_URL is empty (""), the Try It Yourself section
   automatically shows the fallback message + backup video instead
   of a blank iframe.
   ========================================================= */

window.DEMO_CONFIG = {
  // PASTE_GRADIO_LIVE_URL_HERE  <- replace with the live share link from Colab
  GRADIO_LIVE_URL: "",

  // Backup recording shown when the live link is offline.
  // Drop a screen-recording export at assets/demo-walkthrough.mp4
  // (or point this at a hosted URL). Leave "" to hide the video block.
  BACKUP_VIDEO_URL: "assets/demo-walkthrough.mp4",

  // Still image shown if BACKUP_VIDEO_URL is empty or the video file
  // is missing/unreachable, so the fallback never renders broken.
  BACKUP_IMAGE_URL: "assets/images/demo/demo_cell_011_out_00.png",

  // Seconds to wait for the Gradio iframe before falling back.
  IFRAME_LOAD_TIMEOUT_MS: 20000,
};
