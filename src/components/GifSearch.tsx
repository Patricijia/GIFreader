import React, { useEffect, useState } from "react";
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
    const apiKey = "SyBnCaulCHSRBuCc4dAc2VE4lb1HESJf";
    const [gifs, setGifs] = useState<IGifResponse[]>([]);
    const [isPreviousDisabled, setIsPreviousDisabled] = useState(true);
    const [isFeedbackEnabled, setIsFeedbackEnabled] = useState(false);
    const [offsetCount, setOffsetCount] = useState(0);
    const [inputText, setInputText] = useState("");
    const [searchText, setSearchText] = useState("surprise");  
    
    const requestUri = `https://api.giphy.com/v1/gifs/search?api_key=${apiKey}&q=${searchText}&limit=10&offset=${offsetCount}`;
    
    const fetchGifs = async () => {
        const response = await fetch(requestUri);
        const data = await response.json();
        setGifs(data.data as IGifResponse[]);
        console.log('gifsFetched - '+ data.data);
    } // eslint-disable-line react-hooks/exhaustive-deps

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

    return (
    <div className="app">
        <form onSubmit={onSubmit} className="search-form">
        <input className="search-bar" type="text" value={inputText} onChange={e => setInputText(e.target.value)} ></input>
        <button className="search-button" type="submit">Search</button>      
        <button className={!isFeedbackEnabled ? 'show-feedback-btn': 'show-feedback-btn hide-feedback'} onClick={feedbackClicked}>Show Feedback</button>
        <button className={isFeedbackEnabled ? 'show-feedback-btn': 'show-feedback-btn hide-feedback'} onClick={feedbackClicked}>Hide Feedback</button>
        </form>    
        <div className="gifs">
            {gifs.map(each => (              
            <GifDetails key={each.id} title={each.title} imageUri={each.images.original.url} isFeedbackEnabled={isFeedbackEnabled}></GifDetails>
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
    </div>)

}

export default GifSearch;