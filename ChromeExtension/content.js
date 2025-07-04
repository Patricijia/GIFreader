console.log("GIF Accessibility Reader running...");

const seen = new Set();

async function labelGif(gif) {
  const url = gif.src;
  if (seen.has(url)) return;
  seen.add(url);

  try {
    const res = await fetch("http://localhost:8080/describe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });

    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();

    const blip = data.blip?.description || "N/A";
    const custom = data.custom?.description || "N/A";
    const ocr = data.detected_text || "N/A";

    const label = `BLIP: ${blip} | Custom: ${custom} | Text: ${ocr}`;
    gif.alt = label;
    gif.setAttribute("aria-label", label);
    gif.setAttribute("tabindex", "0");

    // Display label visually next to image
    const tag = document.createElement("span");
    tag.innerText = label;
    tag.style.cssText = `
      background: yellow;
      color: black;
      font-size: 12px;
      padding: 2px 4px;
      position: absolute;
      z-index: 9999;
      left: ${gif.getBoundingClientRect().right + window.scrollX + 5}px;
      top: ${gif.getBoundingClientRect().top + window.scrollY}px;
      max-width: 300px;
      display: inline-block;
    `;
    document.body.appendChild(tag);

    console.log("Labeled GIF:", label);
  } catch (err) {
    console.error("GIF Accessibility Error, will retry:", url, err);
    seen.delete(url);
  }
}

function scanAndLabelGIFs() {
  const gifs = Array.from(document.querySelectorAll("img"))
    .filter(img => img.src.includes("giphy") && !seen.has(img.src));

  gifs.forEach(labelGif);
}

scanAndLabelGIFs();
setInterval(scanAndLabelGIFs, 3000);
