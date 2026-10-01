import { Container } from "@cloudflare/containers";

// One long-lived instance of the TrueEdit server (uploads and edits live on its disk).
export class TrueEdit extends Container {
  defaultPort = 8000;
  // keep the instance (and the user's open projects) alive between requests
  sleepAfter = "2h";
  // the app needs no outbound internet: OCR models are baked into the image
  enableInternet = false;

  constructor(ctx, env) {
    super(ctx, env);
    this.envVars = {
      HOST: "0.0.0.0",
      PORT: "8000",
      TRUEEDIT_RETENTION_DAYS: "2",
      // set with:  npx wrangler secret put TRUEEDIT_PASSWORD
      ...(env.TRUEEDIT_PASSWORD ? { TRUEEDIT_PASSWORD: env.TRUEEDIT_PASSWORD } : {}),
    };
  }
}

export default {
  async fetch(request, env) {
    // always the same instance, so a project opened in one request is found by the next
    const stub = env.TRUEEDIT.getByName("main");
    return stub.fetch(request);
  },
};
