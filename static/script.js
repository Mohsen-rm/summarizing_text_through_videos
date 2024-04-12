function transcribeAudio() {
    const audioInput = document.getElementById('audioInput');
    const file = audioInput.files[0];
    const formData = new FormData();
    formData.append('audio', file);

    fetch('/transcribe', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(transcription => {
        document.getElementById('transcription').textContent = transcription;
    })
    .catch(error => console.error('Error:', error));
}
