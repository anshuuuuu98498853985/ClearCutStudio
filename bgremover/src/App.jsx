import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowDownToLine, ArrowLeftRight, Brush, Check, ChevronDown, CircleHelp, CloudOff, Eraser, ImagePlus, LoaderCircle, LockKeyhole, Minus, MousePointer2, Plus, RotateCcw, Sparkles, Upload, X } from 'lucide-react';

const MODELS = {
  quality: { id: 'onnx-community/BiRefNet-ONNX', label: 'BiRefNet · High quality' },
  lite: { id: 'onnx-community/BiRefNet_lite-ONNX', label: 'BiRefNet Lite · Compatibility' },
};
const CACHE = new Map();
const accepted = ['image/jpeg', 'image/png', 'image/webp'];

function formatBytes(bytes) {
  if (!bytes) return '';
  return bytes > 1024 ** 2 ? `${(bytes / 1024 ** 2).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;
}

async function loadModel(modelId, device, dtype, onProgress, transformers) {
  const key = `${modelId}|${device}|${dtype}`;
  if (CACHE.has(key)) return CACHE.get(key);
  const { AutoModel, AutoProcessor } = transformers;
  const promise = Promise.all([
    AutoModel.from_pretrained(modelId, { device, dtype, progress_callback: onProgress }),
    AutoProcessor.from_pretrained(modelId, { progress_callback: onProgress }),
  ]).then(([model, processor]) => ({ model, processor }));
  CACHE.set(key, promise);
  try { return await promise; } catch (error) { CACHE.delete(key); throw error; }
}

function describeError(error) {
  const message = String(error?.message || error || 'Unknown error');
  if (/webgpu|gpu|adapter|shader|device/i.test(message)) return 'This browser or graphics device could not start WebGPU. Try the compatibility option, which runs the lighter model with WebAssembly on your device.';
  if (/fetch|network|download|failed to load|404/i.test(message)) return 'The model could not be downloaded. Check your internet connection and try again. The model files are fetched from Hugging Face and cached by your browser.';
  if (/memory|allocation|out of/i.test(message)) return 'Your browser ran out of memory. Try the lighter model or a smaller image, then close other tabs.';
  if (typeof error === 'number') return `The local AI runtime stopped with code ${error} before returning a mask. Try a current Chrome or Edge browser with WebGPU, or retry the Lite model. No output is available to download.`;
  return `Processing could not start: ${message.slice(0, 240)}`;
}

function makeCanvas(width, height) {
  const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height; return canvas;
}

export default function App() {
  const fileRef = useRef(null); const stageRef = useRef(null); const sourceCanvasRef = useRef(null); const pointerDown = useRef(false); const strokePrev = useRef(null); const workingMask = useRef(null); const sourceUrlRef = useRef(null);
  const [file, setFile] = useState(null); const [sourceUrl, setSourceUrl] = useState(''); const [dims, setDims] = useState(null); const [baseMask, setBaseMask] = useState(null); const [inferredMask, setInferredMask] = useState(null); const [resultUrl, setResultUrl] = useState(''); const [resultBlob, setResultBlob] = useState(null);
  const [modelChoice, setModelChoice] = useState('quality'); const [deviceChoice, setDeviceChoice] = useState('auto'); const [busy, setBusy] = useState(false); const [phase, setPhase] = useState(''); const [progress, setProgress] = useState(null); const [error, setError] = useState(''); const [notice, setNotice] = useState('');
  const [compare, setCompare] = useState(50); const [background, setBackground] = useState('transparent'); const [customColor, setCustomColor] = useState('#dbeafe'); const [zoom, setZoom] = useState(100); const [brushMode, setBrushMode] = useState('erase'); const [brushSize, setBrushSize] = useState(36); const [softness, setSoftness] = useState(1); const [cleanup, setCleanup] = useState(0); const [history, setHistory] = useState([]); const [hasEdits, setHasEdits] = useState(false); const [dragging, setDragging] = useState(false);

  const clearResult = useCallback(() => { setBaseMask(null); setInferredMask(null); setResultBlob(null); setHasEdits(false); workingMask.current = null; setHistory([]); setResultUrl(''); }, []);
  const acceptFile = useCallback(async (nextFile) => {
    setError(''); setNotice('');
    if (!nextFile || !accepted.includes(nextFile.type)) { setError('Choose a JPG, PNG, or WebP image.'); return; }
    if (sourceUrlRef.current) URL.revokeObjectURL(sourceUrlRef.current);
    clearResult();
    const url = URL.createObjectURL(nextFile); sourceUrlRef.current = url;
    const img = new Image(); img.onload = () => {
      setFile(nextFile); setSourceUrl(url); setDims({ width: img.naturalWidth, height: img.naturalHeight }); setCompare(50); setZoom(100);
      const c = makeCanvas(img.naturalWidth, img.naturalHeight); const ctx = c.getContext('2d', { willReadFrequently: true }); ctx.drawImage(img, 0, 0); sourceCanvasRef.current = c;
      if (img.naturalWidth * img.naturalHeight > 32_000_000 || nextFile.size > 25 * 1024 * 1024) setNotice('Large image: processing may use substantial memory. If the tab becomes slow, try a smaller copy.');
    }; img.onerror = () => setError('This image could not be decoded. Try another JPG, PNG, or WebP file.'); img.src = url;
  }, [clearResult]);

  const pasteImage = async () => {
    try {
      if (!navigator.clipboard?.read) throw new Error('Clipboard image access is not available in this browser. Use Ctrl+V or Browse image instead.');
      const items = await navigator.clipboard.read();
      for (const item of items) {
        const type = item.types.find((candidate) => accepted.includes(candidate));
        if (type) { const blob = await item.getType(type); acceptFile(new File([blob], `pasted-image.${type === 'image/jpeg' ? 'jpg' : type.split('/')[1]}`, { type })); return; }
      }
      setError('There is no JPG, PNG, or WebP image in the clipboard. Copy an image and try again.');
    } catch (exception) { setError(exception?.message || 'Could not read an image from the clipboard. Use Ctrl+V or Browse image instead.'); }
  };

  useEffect(() => {
    const onPaste = (event) => { const item = [...(event.clipboardData?.items || [])].find((entry) => accepted.includes(entry.type)); if (item) { const pasted = item.getAsFile(); if (pasted) acceptFile(pasted); } };
    window.addEventListener('paste', onPaste); return () => window.removeEventListener('paste', onPaste);
  }, [acceptFile]);
  useEffect(() => () => { if (sourceUrlRef.current) URL.revokeObjectURL(sourceUrlRef.current); if (resultUrl) URL.revokeObjectURL(resultUrl); }, [resultUrl]);

  const buildOutput = useCallback(async (mask, softnessValue = softness, cleanupValue = cleanup) => {
    if (!file || !dims || !sourceCanvasRef.current || !mask) return;
    const { width, height } = dims; const maskCanvas = makeCanvas(width, height); const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
    const raw = new ImageData(new Uint8ClampedArray(mask), width, height); maskCtx.putImageData(raw, 0, 0);
    if (softnessValue > 0) { const blurred = makeCanvas(width, height); const bctx = blurred.getContext('2d'); bctx.filter = `blur(${softnessValue}px)`; bctx.drawImage(maskCanvas, 0, 0); maskCtx.clearRect(0, 0, width, height); maskCtx.filter = 'none'; maskCtx.drawImage(blurred, 0, 0); }
    const imageData = maskCtx.getImageData(0, 0, width, height); const alpha = imageData.data;
    if (cleanupValue > 0) { const contrast = 1 + cleanupValue / 30; for (let i = 0; i < alpha.length; i += 4) { const v = Math.max(0, Math.min(255, (alpha[i] - 128) * contrast + 128)); alpha[i] = alpha[i + 1] = alpha[i + 2] = v; } maskCtx.putImageData(imageData, 0, 0); }
    const out = makeCanvas(width, height); const octx = out.getContext('2d', { willReadFrequently: true }); octx.drawImage(sourceCanvasRef.current, 0, 0); const pixels = octx.getImageData(0, 0, width, height); const maskPixels = maskCtx.getImageData(0, 0, width, height).data;
    for (let i = 0; i < pixels.data.length; i += 4) pixels.data[i + 3] = Math.round(pixels.data[i + 3] * maskPixels[i] / 255); octx.putImageData(pixels, 0, 0);
    const blob = await new Promise((resolve) => out.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('The browser could not encode the transparent PNG.');
    const url = URL.createObjectURL(blob); setResultUrl((old) => { if (old) URL.revokeObjectURL(old); return url; }); setResultBlob(blob);
  }, [file, dims, softness, cleanup]);

  const runInference = async (forceDevice) => {
    if (!file || busy) return;
    setBusy(true); setError(''); setNotice(''); setProgress(null); clearResult();
    const modelId = MODELS[modelChoice].id;
    try {
      const useWebgpu = forceDevice === 'webgpu' || (forceDevice !== 'wasm' && deviceChoice === 'auto' && !!navigator.gpu);
      const adapter = useWebgpu ? await navigator.gpu.requestAdapter() : null;
      if (useWebgpu && !adapter) throw new Error('No WebGPU adapter is available on this device.');
      const canFp16 = !!adapter?.features?.has('shader-f16');
      const device = useWebgpu ? 'webgpu' : 'wasm'; const dtype = canFp16 ? 'fp16' : 'fp32';
      setPhase('Preparing image…'); const transformers = await import('@huggingface/transformers'); const { RawImage } = transformers; const image = await RawImage.fromBlob(file);
      const { model, processor } = await loadModel(modelId, device, dtype, (info) => {
        if (info?.status === 'progress') { setPhase('Downloading AI model…'); setProgress(Math.max(0, Math.min(100, Math.round(info.progress || 0)))); }
        else if (info?.status === 'done') setProgress(100);
        else if (info?.status === 'initiate') { setPhase('Downloading AI model…'); setProgress(0); }
      }, transformers);
      setPhase('Preparing image for inference…'); const { pixel_values } = await processor(image);
      setPhase('Finding the subject…'); const outputs = await model({ input_image: pixel_values });
      const logits = outputs.output_image;
      if (!logits || !logits.dims || logits.dims.length !== 4 || logits.dims[0] !== 1 || logits.dims[1] !== 1) throw new Error(`Unexpected model output format (${logits?.dims?.join('×') || 'missing dimensions'}). This model version returned an unsupported mask.`);
      const maskTensor = logits[0].sigmoid().mul(255).round().to('uint8');
      const smallMask = RawImage.fromTensor(maskTensor);
      if (smallMask.channels !== 1 || !smallMask.width || !smallMask.height) throw new Error('The model returned an invalid single-channel mask.');
      setPhase('Upscaling the mask to the original size…'); const fullMask = await smallMask.resize(dims.width, dims.height, { resample: 2 });
      const gray = fullMask.grayscale(); const data = new Uint8ClampedArray(dims.width * dims.height * 4);
      for (let i = 0; i < gray.data.length; i++) { const offset = i * 4; data[offset] = data[offset + 1] = data[offset + 2] = gray.data[i]; data[offset + 3] = 255; }
      const mask = data; workingMask.current = new Uint8ClampedArray(mask); setBaseMask(mask); setInferredMask(new Uint8ClampedArray(mask)); setPhase('Finishing transparent PNG…'); await buildOutput(mask); setPhase('Ready'); setNotice(`Background removed · ${dims.width} × ${dims.height}px · ${useWebgpu ? `WebGPU${canFp16 ? ' · FP16' : ''}` : 'WebAssembly'}`);
    } catch (exception) {
      console.error(exception); const problem = describeError(exception); setError(problem);
      setNotice('If this is a WebGPU or memory issue, retry with WebAssembly or choose BiRefNet Lite in the model menu.');
    } finally { setBusy(false); }
  };

  const brushAt = (event) => {
    if (!pointerDown.current || !workingMask.current || !dims || !stageRef.current) return;
    const rect = stageRef.current.getBoundingClientRect(); const scale = Math.min(rect.width / dims.width, rect.height / dims.height) * zoom / 100; const shownW = dims.width * scale; const shownH = dims.height * scale; const left = rect.left + (rect.width - shownW) / 2; const top = rect.top + (rect.height - shownH) / 2;
    const x = Math.round((event.clientX - left) / scale); const y = Math.round((event.clientY - top) / scale); if (x < 0 || y < 0 || x >= dims.width || y >= dims.height) return;
    const updated = new Uint8ClampedArray(workingMask.current); const radius = Math.max(1, brushSize / 2); const value = brushMode === 'erase' ? 0 : 255;
    const prev = strokePrev.current || [x, y]; const span = Math.hypot(x - prev[0], y - prev[1]); const count = Math.max(1, Math.ceil(span / Math.max(1, radius / 3)));
    for (let n = 0; n <= count; n++) { const cx = prev[0] + (x - prev[0]) * n / count; const cy = prev[1] + (y - prev[1]) * n / count;
      for (let py = Math.max(0, Math.floor(cy - radius)); py <= Math.min(dims.height - 1, Math.ceil(cy + radius)); py++) for (let px = Math.max(0, Math.floor(cx - radius)); px <= Math.min(dims.width - 1, Math.ceil(cx + radius)); px++) if ((px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2) updated[(py * dims.width + px) * 4] = updated[(py * dims.width + px) * 4 + 1] = updated[(py * dims.width + px) * 4 + 2] = value;
    }
    strokePrev.current = [x, y]; workingMask.current = updated;
  };
  const startStroke = (event) => { if (!baseMask) return; event.currentTarget.setPointerCapture(event.pointerId); pointerDown.current = true; strokePrev.current = null; setHistory((items) => [...items.slice(-7), new Uint8ClampedArray(workingMask.current)]); brushAt(event); };
  const finishStroke = () => { if (pointerDown.current && workingMask.current) { setBaseMask(new Uint8ClampedArray(workingMask.current)); setHasEdits(true); buildOutput(workingMask.current); } pointerDown.current = false; strokePrev.current = null; };
  const undo = () => { const previous = history.at(-1); if (previous) { setHistory((items) => items.slice(0, -1)); workingMask.current = new Uint8ClampedArray(previous); setBaseMask(new Uint8ClampedArray(previous)); setHasEdits(true); buildOutput(previous); } };
  const resetEdits = async () => { if (inferredMask) { setHistory([]); setHasEdits(false); setSoftness(1); setCleanup(0); workingMask.current = new Uint8ClampedArray(inferredMask); setBaseMask(new Uint8ClampedArray(inferredMask)); await buildOutput(inferredMask, 1, 0); } };
  const changeRefinement = (type, value) => { if (type === 'soft') setSoftness(value); else setCleanup(value); if (workingMask.current) buildOutput(workingMask.current, type === 'soft' ? value : softness, type === 'cleanup' ? value : cleanup); };
  const download = () => { if (!resultBlob || !resultUrl) return; const link = document.createElement('a'); link.href = resultUrl; link.download = 'clearcut-result.png'; document.body.appendChild(link); link.click(); link.remove(); };
  const onDrop = (event) => { event.preventDefault(); setDragging(false); acceptFile(event.dataTransfer.files?.[0]); };
  const backgroundStyle = background === 'transparent' ? undefined : background === 'custom' ? { backgroundColor: customColor } : { backgroundColor: background };

  return <main className="app-shell">
    <header className="topbar"><a className="brand" href="#top" aria-label="ClearCut Studio home"><span className="brand-mark"><Sparkles size={19} strokeWidth={2.3} /></span><span>clearcut<span className="brand-light">studio</span></span></a><nav className="top-nav"><span className="privacy-pill"><LockKeyhole size={14} /> 100% private · runs on your device</span><a href="#how-it-works">How it works</a></nav></header>
    <section className="hero" id="top"><div className="eyebrow"><span className="eyebrow-dot" /> STUDIO QUALITY, RIGHT IN YOUR BROWSER</div><h1>Keep the subject.<br/><span>Lose the background.</span></h1><p className="hero-copy">A little AI magic, a lot more creative freedom. Make your image transparent with on-device AI.</p><div className="hero-proof"><span><Check size={15}/> Always free</span><i/> <span><LockKeyhole size={14}/> Your images stay yours</span><i/> <span>No account needed</span></div></section>
    <section className="workspace" aria-label="Background removal workspace">
      {!file ? <div className={`upload-card ${dragging ? 'dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
        <input ref={fileRef} type="file" accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp" hidden onChange={(event) => { acceptFile(event.target.files?.[0]); event.target.value = ''; }}/>
        <div className="upload-art"><div className="art-sun"/><div className="art-hill hill-back"/><div className="art-hill hill-front"/><span className="art-spark spark-one">✦</span><span className="art-spark spark-two">✧</span><span className="art-frame"><ImagePlus size={20}/></span></div>
        <h2>Start with an image</h2><p>Drag and drop a photo here, or choose one from your device.</p><div className="upload-actions"><button className="button button-primary browse-btn" onClick={() => fileRef.current?.click()}><Upload size={16}/> Browse image</button><button className="button button-secondary paste-btn" onClick={pasteImage}><ImagePlus size={15}/> Paste image</button></div><div className="upload-types">JPG, PNG or WebP <span>·</span> up to 25 MB recommended</div><div className="model-download-note">First AI download: about 490 MB (quality) or 115 MB (Lite) on WebGPU FP16. CPU model files are larger.</div>
        {error && <div className="notice error-notice">{error}<button onClick={() => setError('')} aria-label="Dismiss"><X size={14}/></button></div>}
        <div className="paste-hint"><span className="keycap">Ctrl</span><span className="keycap">V</span><span> to paste an image</span></div>
      </div> : <div className="editor">
        <div className="editor-top"><div className="file-info"><button className="icon-button back-button" onClick={() => { setFile(null); if (sourceUrlRef.current) URL.revokeObjectURL(sourceUrlRef.current); sourceUrlRef.current = null; setSourceUrl(''); setDims(null); clearResult(); setError(''); }} aria-label="Choose another image"><ArrowLeftRight size={16}/></button><div><strong>{file.name}</strong><span>{dims?.width} × {dims?.height}px <i/> {formatBytes(file.size)}</span></div></div><div className="top-actions"><label className="select-wrap"><span className="sr-only">AI model</span><select value={modelChoice} disabled={busy} onChange={(e) => setModelChoice(e.target.value)}><option value="quality">BiRefNet · High quality</option><option value="lite">BiRefNet Lite · Compatibility</option></select><ChevronDown size={14}/></label><button className="icon-button" onClick={() => fileRef.current?.click()} aria-label="Choose another image"><ImagePlus size={17}/></button><input ref={fileRef} type="file" accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp" hidden onChange={(event) => { acceptFile(event.target.files?.[0]); event.target.value = ''; }}/></div></div>
        <div className={`stage ${background === 'transparent' ? 'checker' : ''}`} style={backgroundStyle} ref={stageRef}>
          {sourceUrl && <img className="stage-image" src={resultUrl || sourceUrl} alt={resultUrl ? 'Background removed preview' : 'Image to process'} style={{ transform: `scale(${zoom / 100})` }} draggable="false"/>}
          {resultUrl && <><div className="compare-original" style={{ clipPath: `inset(0 ${100 - compare}% 0 0)` }}><img src={sourceUrl} alt="Original image" style={{ transform: `scale(${zoom / 100})` }} draggable="false"/></div><div className="compare-line" style={{ left: `${compare}%` }}><span><ArrowLeftRight size={15}/></span></div><div className="compare-label label-before">BEFORE</div><div className="compare-label label-after">AFTER</div></>}
          {baseMask && <div className="brush-layer" onPointerDown={startStroke} onPointerMove={brushAt} onPointerUp={finishStroke} onPointerCancel={finishStroke} style={{ cursor: 'crosshair' }} />}
          {!resultUrl && !busy && <div className="stage-empty"><span className="empty-icon"><ImagePlus size={20}/></span><p>Your image is ready when you are</p><button className="button button-primary" onClick={() => runInference()}><Sparkles size={16}/> Remove background</button></div>}
          {busy && <div className="processing-card"><div className="processing-icon"><LoaderCircle size={23} className="spin"/></div><div><strong>{phase || 'Getting things ready…'}</strong><span>{progress != null ? `${progress}% downloaded` : 'This can take a little while on the first run'}</span>{progress != null && <div className="progress-track"><i style={{ width: `${progress}%` }}/></div>}</div></div>}
        </div>
        {resultUrl && <div className="compare-control"><span>Original</span><input aria-label="Compare original and processed image" type="range" min="0" max="100" value={compare} onChange={(event) => setCompare(+event.target.value)} style={{ '--range': `${compare}%` }}/><span>Cutout</span></div>}
        {(error || notice) && <div className={`notice ${error ? 'error-notice' : 'success-notice'}`} role={error ? 'alert' : 'status'}>{error ? <><span>{error}</span><div className="error-actions">{notice && /retry with webassembly/i.test(notice) && <button className="text-button" onClick={() => runInference('wasm')}>Retry with WebAssembly</button>}<button className="text-button" onClick={() => runInference()}>Try again</button></div></> : <><Check size={15}/><span>{notice}</span></>}</div>}
        <div className="editor-bottom"><div className="zoom-controls"><button className="icon-button" onClick={() => setZoom((value) => Math.max(50, value - 10))} aria-label="Zoom out"><Minus size={15}/></button><button className="zoom-value" onClick={() => setZoom(100)}>{zoom}%</button><button className="icon-button" onClick={() => setZoom((value) => Math.min(200, value + 10))} aria-label="Zoom in"><Plus size={15}/></button><button className="fit-button" onClick={() => setZoom(100)}><MousePointer2 size={14}/> Fit</button></div><div className="bottom-actions"><button className="button button-secondary" disabled={busy} onClick={() => runInference()}><Sparkles size={15}/> {resultUrl ? 'Run again' : 'Remove background'}</button><button className="button button-download" disabled={!resultBlob || busy} onClick={download}><ArrowDownToLine size={16}/> Download PNG <span>Free</span></button></div></div>
      </div>}
    </section>
    {file && <section className="tools-panel" aria-label="Image editing tools"><div className="tools-heading"><div><span className="panel-kicker">MAKE IT YOURS</span><h2>Fine-tune your cutout</h2></div><button className="text-button" onClick={resetEdits} disabled={!baseMask || busy}><RotateCcw size={14}/> Reset edits</button></div><div className="tools-grid">
      <div className="tool-card brush-card"><div className="tool-title"><span className="tool-icon"><Brush size={16}/></span><div><strong>Mask brush</strong><small>Paint areas to keep or remove</small></div><button className="undo-button" onClick={undo} disabled={!history.length} title="Undo"><RotateCcw size={15}/></button></div><div className="brush-toggle"><button className={brushMode === 'erase' ? 'active' : ''} onClick={() => setBrushMode('erase')} disabled={!baseMask}><Eraser size={15}/> Erase</button><button className={brushMode === 'restore' ? 'active' : ''} onClick={() => setBrushMode('restore')} disabled={!baseMask}><Brush size={15}/> Restore</button></div><label className="range-label"><span>Brush size</span><span>{brushSize}px</span><input type="range" min="8" max="120" step="2" value={brushSize} onChange={(e) => setBrushSize(+e.target.value)}/></label></div>
      <div className="tool-card"><div className="tool-title"><span className="tool-icon"><Sparkles size={16}/></span><div><strong>Edge refinement</strong><small>Softness and edge cleanup alter PNG alpha</small></div></div><label className="range-label"><span>Mask softness</span><span>{softness}px</span><input type="range" min="0" max="6" step="0.5" value={softness} onChange={(e) => changeRefinement('soft', +e.target.value)} disabled={!baseMask}/></label><label className="range-label"><span>Edge cleanup</span><span>{cleanup}%</span><input type="range" min="0" max="60" step="1" value={cleanup} onChange={(e) => changeRefinement('cleanup', +e.target.value)} disabled={!baseMask}/></label></div>
      <div className="tool-card background-card"><div className="tool-title"><span className="tool-icon"><ImagePlus size={16}/></span><div><strong>Preview background</strong><small>Preview only · download stays transparent</small></div></div><div className="swatches"><button aria-label="Transparent background" className={`swatch swatch-check ${background === 'transparent' ? 'selected' : ''}`} onClick={() => setBackground('transparent')}><span/></button><button aria-label="White background" className={`swatch white ${background === 'white' ? 'selected' : ''}`} onClick={() => setBackground('white')}/><button aria-label="Black background" className={`swatch black ${background === 'black' ? 'selected' : ''}`} onClick={() => setBackground('black')}/><label className={`swatch custom ${background === 'custom' ? 'selected' : ''}`} title="Choose a color" style={{ backgroundColor: customColor }}><input type="color" value={customColor} onChange={(e) => { setCustomColor(e.target.value); setBackground('custom'); }}/></label><span className="swatch-caption">Solid color</span></div></div>
    </div>{hasEdits && <div className="edit-note"><Check size={14}/> Brush changes are applied to the exported alpha mask.</div>}</section>}
    {!file && <section className="benefits"><article><span><Sparkles size={17}/></span><div><h3>Looks like magic</h3><p>BiRefNet AI finds your subject and creates a clean, detailed mask.</p></div></article><article><span><LockKeyhole size={17}/></span><div><h3>Private by design</h3><p>Your photo never leaves this browser. Processing happens on your device.</p></div></article><article><span><CloudOff size={17}/></span><div><h3>Free, for real</h3><p>No account, no credits, no watermark. Just your finished PNG.</p></div></article></section>}
    <section className="how" id="how-it-works"><div className="how-label"><CircleHelp size={15}/> A QUICK NOTE</div><p>The first time you remove a background, ClearCut downloads the AI model from Hugging Face (about 490 MB for quality FP16 or 115 MB for Lite FP16; CPU weights are larger). It’s a one-time download per model for this browser, and then the model can be reused from your browser cache. Your image itself never gets uploaded.</p></section>
    <footer><a className="brand footer-brand" href="#top"><span className="brand-mark"><Sparkles size={15}/></span><span>clearcut<span className="brand-light">studio</span></span></a><span>Thoughtful tools. Your images, always yours.</span><a href="#how-it-works">Privacy & model info</a></footer>
  </main>;
}
