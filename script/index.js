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
    try {
        const result = new Date(timeInSeconds * 1000).toISOString().substr(11, 8);
        return {
            minutes: result.substr(3, 2),
            seconds: result.substr(6, 2),
        };
    }
    catch (e) {
        console.log('wrong time format');
        return {
            minutes: 'nan',
            seconds: 'nan',
        };
    }
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
    duration.setAttribute('datetime', `${time.minutes}m ${time.seconds}s`);
    video.playbackRate = 0.85;
}
video.addEventListener('loadedmetadata', updateVideoInfo);
// Progress bar END

// Update function BEGIN
function updateTimeElapsed() {
    const time = formatTime(Math.round(video.currentTime));
    timeElapsed.innerText = `${time.minutes}:${time.seconds}`;
    timeElapsed.setAttribute('datetime', `${time.minutes}m ${time.seconds}s`)
}

function updateProgress() {
    seek.value = Math.round(video.currentTime);
    progressBar.value = Math.round(video.currentTime);
}

function updateEverything() {
    updateVideoInfo();
    updateTimeElapsed();
    updateProgress();
}

setInterval(updateEverything, 500);
// video.addEventListener('timeupdate', updateEverything);
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

    volumeButton.setAttribute('data-title', 'Mute')

    if (video.muted || video.volume === 0) {
        volumeMute.classList.remove('hidden');
        volumeButton.setAttribute('data-title', 'Unmute')
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

var server_ip, flask_port;
$.getJSON('config.json', function (json) {
    server_ip = json.server_ip;
    flask_port = json.flask_port;
});

function sendRequest(method, url) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open(method, url);
    xmlHttp.send(null);
    return xmlHttp.responseText;
}

// Shutdown START
const shutdownButton = document.getElementById('shutdown-button');
function shutdown() {
    sendRequest('GET', `http://${server_ip}:${flask_port}/shutdown`);
}
shutdownButton.addEventListener('click', shutdown);
// Shutdown END

// Deselect START
const deselectButton = document.getElementById('deselect-button');
function deselect() {
    sendRequest('GET', `http://${server_ip}:${flask_port}/data?coor=deselect`);
}
deselectButton.addEventListener('click', deselect);
// Deselect END

// Click event START
function getClickCoordinate(event) {
    var x = event.clientX + window.pageXOffset - videoContainer.offsetLeft;
    var y = event.clientY + window.pageYOffset - videoContainer.offsetTop;
    var height = video.offsetHeight;
    var width = video.offsetWidth;
    x = x < 0 ? 0 : x;
    y = y < 0 ? 0 : y;
    sendRequest('GET', `http://${server_ip}:${flask_port}/data?coor=`+x+','+y+','+height+','+width);
    event.preventDefault();
}
video.addEventListener('click', getClickCoordinate);
// Click event END
