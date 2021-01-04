// UI BEGIN
const video = document.getElementById('video');
const videoControls = document.getElementById('video-controls');
const videoContainer = document.getElementById('video-container');

const videoWorks = !!document.createElement('video').canPlayType;
if (videoWorks) {
    video.controls = false;
    videoControls.classList.remove('hidden');
}

// Play/Pause BEGIN
const playButton = document.getElementById('play');
function togglePlay() {
    if (video.paused || video.ended) {
        video.play();
    } else {
        video.pause();
    }
}
playButton.addEventListener('click', togglePlay);

const playbackIcons = document.querySelectorAll('.playback-icons use');
function updatePlayButton() {
    playbackIcons.forEach(icon => icon.classList.toggle('hidden'));

    if (video.paused) {
        playButton.setAttribute('data-title', 'Play')
    } else {
        playButton.setAttribute('data-title', 'Pause')
    }
}
video.addEventListener('play', updatePlayButton);
video.addEventListener('pause', updatePlayButton);
// Play/Pause END

// Duration/Time elapsed BEGIN
const timeElapsed = document.getElementById('time-elapsed');
const duration = document.getElementById('duration');
function formatTime(timeInSeconds) {
    const result = new Date(timeInSeconds * 1000).toISOString().substr(11, 8);
    return {
        minutes: result.substr(3, 2),
        seconds: result.substr(6, 2),
    };
};
// Duration/Time elapsed END

// Progress bar BEGIN
const progressBar = document.getElementById('progress-bar');
const seek = document.getElementById('seek');
function updateVideoInfo() {
    const videoDuration = Math.round(video.duration);
    seek.setAttribute('max', videoDuration);
    progressBar.setAttribute('max', videoDuration);
    const time = formatTime(videoDuration);
    duration.innerText = `${time.minutes}:${time.seconds}`;
    duration.setAttribute('datetime', `${time.minutes}m ${time.seconds}s`)
}
video.addEventListener('loadedmetadata', updateVideoInfo);
// Progress bar END

// Update function BEGIN
function updateTimeElapsed() {
    updateVideoInfo();
    const time = formatTime(Math.round(video.currentTime));
    timeElapsed.innerText = `${time.minutes}:${time.seconds}`;
    timeElapsed.setAttribute('datetime', `${time.minutes}m ${time.seconds}s`)
}
video.addEventListener('timeupdate', updateTimeElapsed);

function updateProgress() {
    seek.value = Math.floor(video.currentTime);
    progressBar.value = Math.floor(video.currentTime);
}
video.addEventListener('timeupdate', updateProgress);
// Update function END

const seekTooltip = document.getElementById('seek-tooltip');
function updateSeekTooltip(event) {
    const skipTo = Math.round((event.offsetX / event.target.clientWidth) * parseInt(event.target.getAttribute('max'), 10));
    seek.setAttribute('data-seek', skipTo)
    const t = formatTime(skipTo);
    seekTooltip.textContent = `${t.minutes}:${t.seconds}`;
    const rect = video.getBoundingClientRect();
    seekTooltip.style.left = `${event.pageX - rect.left}px`;
}
seek.addEventListener('mousemove', updateSeekTooltip);

function skipAhead(event) {
    const skipTo = event.target.dataset.seek ? event.target.dataset.seek : event.target.value;
    video.currentTime = skipTo;
    progressBar.value = skipTo;
    seek.value = skipTo;
}
seek.addEventListener('input', skipAhead);

// Volume control BEGIN
const volumeButton = document.getElementById('volume-button');
const volumeIcons = document.querySelectorAll('.volume-button use');
const volumeMute = document.querySelector('use[href="#volume-mute"]');
const volumeLow = document.querySelector('use[href="#volume-low"]');
const volumeHigh = document.querySelector('use[href="#volume-high"]');
const volume = document.getElementById('volume');

function updateVolume() {
    if (video.muted) {
        video.muted = false;
    }
    video.volume = volume.value;
}
volume.addEventListener('input', updateVolume);

function updateVolumeIcon() {
    volumeIcons.forEach(icon => {
        icon.classList.add('hidden');
    });
    
    volumeButton.setAttribute('data-title', 'Mute (m)')
    
    if (video.muted || video.volume === 0) {
        volumeMute.classList.remove('hidden');
        volumeButton.setAttribute('data-title', 'Unmute (m)')
    } else if (video.volume > 0 && video.volume <= 0.5) {
        volumeLow.classList.remove('hidden');
    } else {
        volumeHigh.classList.remove('hidden');
    }
}
video.addEventListener('volumechange', updateVolumeIcon);

function toggleMute() {
    video.muted = !video.muted;
    if (video.muted) {
        volume.setAttribute('data-volume', volume.value);
        volume.value = 0;
    } else {
        volume.value = volume.dataset.volume;
    }
}
volumeButton.addEventListener('click', toggleMute);
// Volume control END
// UI END

// Connect to WebSocket server
var ws = new WebSocket('ws://127.0.0.1:3333');
ws.onopen = function () {
    console.log('Connected to ws server');
};

function sendMessage(msg) {
    ws.send(msg);
};

// Click event
function getClickCoordinate(event) {
    var x = event.clientX + window.pageXOffset - videoContainer.offsetLeft;
    var y = event.clientY + window.pageYOffset - videoContainer.offsetTop;
    // const videoBoundingBox = video.getBoundingClientRect();
    // var x = event.clientX + window.pageXOffset - Math.round(videoBoundingBox.left);
    // var y = event.clientY + window.pageYOffset - Math.round(videoBoundingBox.top);
    x = x < 0 ? 0 : x;
    y = y < 0 ? 0 : y;
    sendMessage(x + ',' + y);
    event.preventDefault();
}
video.addEventListener('click', getClickCoordinate);