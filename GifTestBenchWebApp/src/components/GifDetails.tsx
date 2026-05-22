import React, { useState, useRef, useEffect } from "react";
import './GifDetails.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faThumbsUp, faThumbsDown } from '@fortawesome/free-solid-svg-icons';

interface IGifDetailProps {
    title: string,
    imageUri: string
    isFeedbackEnabled: boolean
    onCaptioned?: () => void
}

function GifDetails({title, imageUri, isFeedbackEnabled, onCaptioned}: IGifDetailProps) {
    const [isSelected, setIsSelected] = useState(false);
    const [isSelectedDislike, setIsSelectedDislike] = useState(false);
    const imgRef = useRef<HTMLImageElement | null>(null);
    const firedRef = useRef(false);

    // Wait for the CNN-LSTM extension to set aria-label on this <img>.
    // That's our signal that the GIF has been captioned, not just loaded.
    useEffect(() => {
        const img = imgRef.current;
        if (!img) return;
        firedRef.current = false;

        const fireOnce = () => {
            if (firedRef.current) return;
            firedRef.current = true;
            onCaptioned && onCaptioned();
        };

        // If the extension already labeled it before we mounted, fire now.
        if (img.getAttribute("aria-label")) {
            fireOnce();
            return;
        }

        const observer = new MutationObserver(() => {
            if (img.getAttribute("aria-label")) {
                fireOnce();
                observer.disconnect();
            }
        });
        observer.observe(img, { attributes: true, attributeFilter: ["aria-label"] });

        return () => observer.disconnect();
    }, [imageUri, onCaptioned]);

    const speakCaption = () => {
        const img = imgRef.current;
        if (!img || !("speechSynthesis" in window)) return;
        const text = img.getAttribute("aria-label") || img.alt || "no caption";
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
    };

    const onClick = (e: any) => {
        e.preventDefault();
        setIsSelected(!isSelected);
    }

    const onClickFalse = (e: any) => {
        e.preventDefault();
        setIsSelectedDislike(!isSelectedDislike);
    }

    return (
        <div className="gif" >
                <img
                    ref={imgRef}
                    className="marginTop"
                    tabIndex={0}
                    src={imageUri}
                    alt={title}
                    onClick={speakCaption}
                    onFocus={speakCaption}
                    style={{ cursor: "pointer" }}
                ></img>
                <div className={isFeedbackEnabled ? '' : 'hide-feedback'}>
                    <button className="buttons" aria-label="Is this a correct description, thumbs up" onClick={onClick} ><FontAwesomeIcon className={isSelected ? 'selected-icon feedback-icon' : 'feedback-icon'} icon={faThumbsUp} /></button>
                    <button className="buttons" aria-label="Is this an incorrect description, thumbs down" onClick={onClickFalse} ><FontAwesomeIcon className={isSelectedDislike ? 'selected-icon-false feedback-icon' : 'feedback-icon'} icon={faThumbsDown}  /></button>
                </div>
        </div>
    );
}

export default GifDetails;
