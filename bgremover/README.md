# ClearCut Studio

ClearCut Studio is an experimental, browser-based image background-removal project built with React and Vite. Its interface is designed to load an image, run a BiRefNet model with Transformers.js, preview and adjust a mask, and export a transparent PNG.

> **Status: experimental. AI background removal has NOT yet been verified as working end to end.** In the available browser environment, the model inference attempt failed before producing a mask. Treat the application as work in progress; do not rely on its output for production use.

## Current status and limitations

- The interface and image selection flow are implemented. Successful AI inference and a valid transparent PNG export have not been confirmed.
- The project targets in-browser inference with a BiRefNet model. First use requires downloading model and runtime assets; the high-quality model weights can be hundreds of megabytes. Inference depends on browser, graphics hardware, memory, and runtime support.
- The project has no server-side inference service. Image privacy and network behavior should be independently reviewed before using sensitive images: model/runtime assets are fetched from external providers, and this project has not completed a privacy audit.
- Large images may use substantial browser memory. Mask editing is session-only and is not saved as a project file.
- There is no automated test suite or verified cross-browser support documented yet.

## Windows development setup

1. Install a current Node.js LTS release from [nodejs.org](https://nodejs.org/).
2. Open PowerShell in the project folder.
3. Install pnpm if it is not already available, then install the dependencies:

   ```powershell
   npm.cmd install --global pnpm
   pnpm.cmd install
   ```

   The repository includes `pnpm-lock.yaml` to pin dependency versions.
4. Start the development server:

   ```powershell
   pnpm.cmd dev
   ```

5. Open the local URL printed by Vite, usually `http://localhost:5173`.

The first inference attempt may download large model files. Avoid starting it on a metered or low-disk connection. To create a production build, run `pnpm.cmd build`; build output is written to the ignored `dist/` folder. A successful build does not verify AI inference.

## Project structure

- `src/App.jsx` — application interface and image-processing flow
- `src/styles.css` — application styles
- `src/main.jsx` — React entry point
- `index.html` — Vite document entry point
- `vite.config.js` — Vite configuration
- `package.json` and `pnpm-lock.yaml` — dependencies and lockfile

## Model and attribution

The code is configured to use BiRefNet ONNX models through Transformers.js. Model files are not included in this repository; they are fetched when needed. Review the model repository and its license before use or redistribution: [onnx-community/BiRefNet-ONNX](https://huggingface.co/onnx-community/BiRefNet-ONNX).

## Verification

The Vite production build completed in the development environment, and the page loaded with a synthetic PNG. AI inference did **not** complete there: WebGPU was unavailable, and the WebAssembly attempt stopped with runtime code `240595976` before producing a mask. Therefore successful background removal and PNG export remain unverified. This result may differ on other machines, but no working result is claimed here.
