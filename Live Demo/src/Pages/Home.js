import { useState, useRef, useEffect } from 'react'
import { isMobile } from "react-device-detect";
import PageTemplate from './Template'
import { CircularProgress } from '@mui/material';
import Webcam from 'react-webcam'
import axios from 'axios'
import FormData from 'form-data'
import '../css/Home.css';

function App() {
    const webcamRef = useRef(null);
    const [data, setData] = useState(null)

    useEffect(() => {
        async function callPost() {
            //Image Logic
            let im = webcamRef.current.getScreenshot()
            let formData = new FormData()
            if (!im) return setTimeout(callPost, 1000)
            var block = im.split(";");
            var contentType = block[0].split(":")[1];
            var realData = block[1].split(",")[1];
            var blob = b64toBlob(realData, contentType);
            if (!blob) return setTimeout(callPost, 1000)
            formData.append('image', blob)

            //Model selection
            let choice = document.getElementById('model')
            formData.append('model', choice.value)

            //Post
            const response = await axios({
                method: 'post',
                url: 'http://a.vibot.tech:5972/predict',
                mode: 'cors',
                headers: {
                    'Access-Control-Allow-Origin': '*'
                },
                data: formData
            }).catch(er => { })

            //return data
            if (!response || !response.data) return setTimeout(callPost, 1000)
            const data = response.data
            setData(data);
            setTimeout(callPost, 1000)
        }
        callPost()
    }, [])


    return (
        <>
            <PageTemplate highLight="0" />
            <div className='Camera'>
                <div>
                    <label>Choose a model: </label>
                    <select id='model'>
                        <option value='cnn'>CNN</option>
                        <option value='groupedcl'>Grouped Convolution Layers</option>
                    </select>
                </div>
                <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    videoConstraints={(isMobile) ? { videoConstraints } : undefined}
                    style={{ maxHeight: '85vh', maxWidth: '85vw', position: 'relative' }}
                />
            </div>
            {data ?
                <div className='PredictionArea'>
                    <h1>Predicted Speed: {data.pred ? data.pred : 'Error'}</h1>
                </div>
                :
                <CircularProgress />}
        </>
    )
}

export default App;


const videoConstraints = {
    facingMode: { exact: "environment" }
};

function b64toBlob(b64Data, contentType, sliceSize) {
    contentType = contentType || '';
    sliceSize = sliceSize || 512;

    var byteCharacters = atob(b64Data);
    var byteArrays = [];

    for (var offset = 0; offset < byteCharacters.length; offset += sliceSize) {
        var slice = byteCharacters.slice(offset, offset + sliceSize);

        var byteNumbers = new Array(slice.length);
        for (var i = 0; i < slice.length; i++) {
            byteNumbers[i] = slice.charCodeAt(i);
        }

        var byteArray = new Uint8Array(byteNumbers);

        byteArrays.push(byteArray);
    }

    var blob = new Blob(byteArrays, { type: contentType });
    return blob;
}
