import { useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowRight, faArrowLeft } from '@fortawesome/free-solid-svg-icons';
import GifDetails from "./GifDetails";

interface IGifResponse {
    title: string,
    url: string,
    id: string,
    embed_url: string,
    images: any
}

function GifSearch() {
    const apiKey = "omoTpuWA3wGJxSa4MNPE54Tl6UGrEPbU";
    const [gifs, setGifs] = useState<IGifResponse[]>([]);
    const [isPreviousDisabled, setIsPreviousDisabled] = useState(true);
    const [isFeedbackEnabled, setIsFeedbackEnabled] = useState(false);
    const [offsetCount, setOffsetCount] = useState(0);
    const [inputText, setInputText] = useState("");
    const [searchText, setSearchText] = useState("surprise");
    const [loadedCount, setLoadedCount] = useState(0);
    const [statusMsg, setStatusMsg] = useState("");
    const [showReady, setShowReady] = useState(false);
    const [startTime, setStartTime] = useState<number>(0);
    const [elapsedMs, setElapsedMs] = useState<number>(0);
    const [voiceStarted, setVoiceStarted] = useState(false);
    const pendingTextRef = useRef<string>("");

    const requestUri = `https://api.giphy.com/v1/gifs/search?api_key=${apiKey}&q=${searchText}&limit=10&offset=${offsetCount}`;

    const fetchGifs = async () => {
        setLoadedCount(0);
        setStatusMsg("");   // stay silent until ready — don't make Orca speak mid-load
        setShowReady(false);
        setElapsedMs(0);
        setVoiceStarted(false);
        setStartTime(performance.now());
        const response = await fetch(requestUri);
        const data = await response.json();
        setGifs(data.data as IGifResponse[]);
        console.log('gifsFetched - '+ data.data);
    } // eslint-disable-line react-hooks/exhaustive-deps

    const handleGifLoaded = () => {
        setLoadedCount(c => c + 1);
    };

    // Live ticking elapsed time. Stops once the live region is populated
    // (the moment Orca has the captions to read).
    useEffect(() => {
        if (!startTime) return;
        if (voiceStarted) return;
        const interval = setInterval(() => {
            setElapsedMs(performance.now() - startTime);
        }, 100);
        return () => clearInterval(interval);
    }, [startTime, voiceStarted]);

    // Hide the entire app from the screen reader until GIFs are fully loaded.
    // We toggle aria-hidden on #root and render the alert/loading toast via a
    // portal to <body> so they remain accessible to AT.
    useEffect(() => {
        const root = document.getElementById("root");
        if (!root) return;
        const ready = gifs.length > 0 && loadedCount >= gifs.length;
        if (ready) {
            root.removeAttribute("aria-hidden");
            root.removeAttribute("inert");
        } else {
            root.setAttribute("aria-hidden", "true");
            root.setAttribute("inert", "");
        }
    }, [loadedCount, gifs]);

    useEffect(() => {
        if (gifs.length > 0 && loadedCount >= gifs.length && !voiceStarted) {
            setShowReady(true);

            const announceTimer = setTimeout(() => {
                const imgs = Array.from(document.querySelectorAll<HTMLImageElement>(".gifs img"));
                const captions = imgs
                    .map((img, i) => `GIF ${i + 1}: ${img.getAttribute("aria-label") || img.alt || "no caption"}.`)
                    .join(" ");

                if (imgs[0]) imgs[0].focus({ preventScroll: false });

                // Build the full announcement. Orca will read it via the
                // polite live region below.
                const ms = startTime ? Math.round(performance.now() - startTime) : 0;
                const seconds = (ms / 1000).toFixed(1);
                const text = `Accessibility ready in ${seconds} seconds. ` +
                             `All ${gifs.length} GIFs captioned. ${captions}`;
                setStatusMsg(text);
                setElapsedMs(ms);
                pendingTextRef.current = text;
                speakPending();
            }, 50);

            const hideTimer = setTimeout(() => setShowReady(false), 8000);
            return () => { clearTimeout(announceTimer); clearTimeout(hideTimer); };
        }
    }, [loadedCount, gifs, startTime, voiceStarted]);

    useEffect(() => {
        fetchGifs()
        // eslint-disable-next-line
    }, [searchText]);

    useEffect(() => {
        fetchGifs()
        // eslint-disable-next-line
    }, [offsetCount]);

    const onSubmit = (e: any) => {
        e.preventDefault();
        setOffsetCount(0);
        setIsPreviousDisabled(true);
        setSearchText(inputText);        
        console.log(inputText);
    }

    const getPreviousData = () =>
    {
        if(offsetCount > 0) 
        {
            setOffsetCount(offsetCount-10);
            fetchGifs();
        }
        
        if(offsetCount === 0)
        {
            setIsPreviousDisabled(true);
        }
    };

    const getNextData = () =>
    {
        setIsPreviousDisabled(false);
        setOffsetCount(offsetCount+10);
        fetchGifs();
    };

    const feedbackClicked = (e: any) =>
    {
        e.preventDefault();
        setIsFeedbackEnabled(!isFeedbackEnabled);        
    };

    const isFullyLoaded = gifs.length > 0 && loadedCount >= gifs.length;

    // Speak the pending announcement via the browser's Web Speech API.
    // Routes through speech-dispatcher / espeak-ng on Linux. Some browsers
    // block speech until a user gesture; if that happens, the click handler
    // below will retry on the first interaction.
    const speakPending = () => {
        const text = pendingTextRef.current;
        if (!text || !("speechSynthesis" in window)) return;
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = 1.05;
        utter.onstart = () => {
            const ms = startTime ? Math.round(performance.now() - startTime) : 0;
            setElapsedMs(ms);
            setVoiceStarted(true);
        };
        window.speechSynthesis.speak(utter);
    };

    // First user click re-tries pending speech in case autoplay was blocked.
    useEffect(() => {
        const unlock = () => {
            if (!voiceStarted && pendingTextRef.current) speakPending();
        };
        document.addEventListener("click", unlock);
        return () => document.removeEventListener("click", unlock);
        // eslint-disable-next-line
    }, [voiceStarted, startTime]);

    // Toasts must live OUTSIDE #root, because #root is aria-hidden during load.
    // We render the long announcement (statusMsg) into a polite live region
    // that's also in the portal — keeping it outside #root means it can never
    // be aria-hidden when the captions come in.
    const toasts = (
        <>
            {/* Polite announcement: Orca reads this once it changes from "" to the
                long captions string. Lives outside #root so aria-hidden can't gag it. */}
            <div
                role="status"
                aria-live="polite"
                style={{
                    position: "absolute",
                    width: "1px",
                    height: "1px",
                    margin: "-1px",
                    padding: 0,
                    overflow: "hidden",
                    clip: "rect(0 0 0 0)",
                    whiteSpace: "nowrap",
                    border: 0,
                }}
            >
                {statusMsg}
            </div>
            {showReady && (
                <div
                    onClick={() => setShowReady(false)}
                    style={{
                        position: "fixed",
                        top: "20px",
                        left: "50%",
                        transform: "translateX(-50%)",
                        background: "#1f7a3f",
                        color: "#fff",
                        padding: "14px 22px",
                        borderRadius: "8px",
                        fontSize: "16px",
                        fontWeight: 600,
                        boxShadow: "0 6px 20px rgba(0,0,0,0.25)",
                        cursor: "pointer",
                        zIndex: 9999,
                        display: "flex",
                        alignItems: "center",
                        gap: "10px",
                    }}
                    aria-hidden="true"
                >
                    <span>✓</span>
                    <span>Accessibility ready — all GIFs captioned</span>
                    <span
                        style={{
                            background: "rgba(0,0,0,0.25)",
                            padding: "3px 8px",
                            borderRadius: "12px",
                            fontSize: "13px",
                            fontVariantNumeric: "tabular-nums",
                            marginLeft: "4px",
                        }}
                    >
                        ⏱ {(elapsedMs / 1000).toFixed(1)}s
                    </span>
                </div>
            )}
            {!showReady && gifs.length > 0 && loadedCount < gifs.length && (
                <div
                    aria-hidden="true"
                    style={{
                        position: "fixed",
                        top: "20px",
                        left: "50%",
                        transform: "translateX(-50%)",
                        background: "#444",
                        color: "#fff",
                        padding: "12px 20px",
                        borderRadius: "8px",
                        fontSize: "15px",
                        boxShadow: "0 4px 12px rgba(0,0,0,0.25)",
                        zIndex: 9998,
                    }}
                >
                    Captioning GIFs… {loadedCount}/{gifs.length} · {(elapsedMs / 1000).toFixed(1)}s
                </div>
            )}
        </>
    );

    return (
    <div className="app">
        {ReactDOM.createPortal(toasts, document.body)}
        {/* Everything below is hidden from screen reader (and keyboard) until
            all GIFs have finished loading. This prevents Orca from reading
            partial / uncaptioned content. */}
        <div
            aria-hidden={!isFullyLoaded}
            {...(!isFullyLoaded ? { inert: "" as any } : {})}
        >
            <form onSubmit={onSubmit} className="search-form">
                <input className="search-bar" type="text" value={inputText} onChange={e => setInputText(e.target.value)} ></input>
                <button className="search-button" type="submit">Search</button>
                <button className={!isFeedbackEnabled ? 'show-feedback-btn': 'show-feedback-btn hide-feedback'} onClick={feedbackClicked}>Show Feedback</button>
                <button className={isFeedbackEnabled ? 'show-feedback-btn': 'show-feedback-btn hide-feedback'} onClick={feedbackClicked}>Hide Feedback</button>
            </form>
            <div className="gifs">
                {gifs.map(each => (
                <GifDetails
                    key={each.id}
                    title={each.title}
                    imageUri={each.images.original.url}
                    isFeedbackEnabled={isFeedbackEnabled}
                    onCaptioned={handleGifLoaded}
                ></GifDetails>
                ))}
            </div>
            <div className="next-previous-buttons">
                <div className="previous-button">
                    <button className="buttons" onClick={e => getPreviousData()} disabled={isPreviousDisabled}><FontAwesomeIcon icon={faArrowLeft} />
                    <p>PREV</p>
                    </button>
                </div>
                <div className="next-button">
                    <button className="buttons" onClick={e => getNextData()}><FontAwesomeIcon icon={faArrowRight} />
                    <p>NEXT</p>
                    </button>
                </div>
            </div>
        </div>
    </div>)

}

export default GifSearch;