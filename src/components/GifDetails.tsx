import React, { useState } from "react";
import './GifDetails.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faThumbsUp, faThumbsDown } from '@fortawesome/free-solid-svg-icons';

interface IGifDetailProps {
    title: string,
    imageUri: string    
    isFeedbackEnabled: boolean
}

function GifDetails({title, imageUri, isFeedbackEnabled}: IGifDetailProps) {
    const [isSelected, setIsSelected] = useState(false);
    const [isSelectedDislike, setIsSelectedDislike] = useState(false);
    const onClick = (e: any) => {         
        e.preventDefault();        
        console.log(isSelected);
        setIsSelected(!isSelected);
    }

    const onClickFalse = (e: any) => {        
        e.preventDefault();         
        console.log(isSelectedDislike);
        setIsSelectedDislike(!isSelectedDislike);
    }

    return (
        <div className="gif" >
                {/* <h3>{title}</h3> */}
                <img className="marginTop" tabIndex={0} src={imageUri} alt ='' ></img>
                <div className={isFeedbackEnabled ? '' : 'hide-feedback'}>
                    <button className="buttons" aria-label="Is this a correct description, thumbs up" onClick={onClick} ><FontAwesomeIcon className={isSelected ? 'selected-icon feedback-icon' : 'feedback-icon'} icon={faThumbsUp} /></button>
                    <button className="buttons" aria-label="Is this an incorrect description, thumbs down" onClick={onClickFalse} ><FontAwesomeIcon className={isSelectedDislike ? 'selected-icon-false feedback-icon' : 'feedback-icon'} icon={faThumbsDown}  /></button>
                </div>
        </div>
    );
}

export default GifDetails;