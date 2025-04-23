const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs');
const { spawn } = require('child_process');

const SITE_URL = process.env.SITE_URL;

class fetchVideos {
  constructor(html) {
    this.html = html;
    this.videos = [];
  }

  async fetchVideos() {
    try {
      // Parse HTML with cheerio
      const $ = cheerio.load(this.html);
      
      // Create an array to hold all download promises
      const downloadPromises = [];
      
      // Find all elements with data-mediumthumb attribute
      $('[data-mediumthumb]').each((i, element) => {
        const thumbnailUrl = $(element).attr('data-mediumthumb');
        const mediabook = $(element).attr('data-mediabook');
        const newVideo = {
          images: []
        };

        const processVideoPromise = (async () => {
          try {
            // Download thumbnail if available
            if (thumbnailUrl && thumbnailUrl.startsWith('http')) {
              await this.downloadThumbnail(thumbnailUrl, newVideo);
            }

            // Download and process video if available
            if (mediabook && mediabook.startsWith('http')) {
              await this.processVideo(mediabook, newVideo);
            }

            // Only add videos with images
            if (newVideo.images.length > 0) {
              this.videos.push(newVideo);
            }
          } catch (err) {
            console.error('Error processing media element:', err);
          }
        })();

        // Add this promise to our array of promises to track
        downloadPromises.push(processVideoPromise);
      });

      // Wait for all downloads to complete
      await Promise.all(downloadPromises);
      
      console.log('Videos fetched successfully:', this.videos);
      return this.videos;
    } catch (error) {
      console.error('Error fetching videos:', error);
      throw error;
    }
  }

  async downloadThumbnail(url, videoObj) {
    try {
      const imageId = Math.random().toString(36).substring(2, 15);
      const imagePath = `/tmp/${imageId}.jpg`;
      
      const response = await axios({
        method: 'GET',
        url: url,
        responseType: 'stream'
      });
      
      await new Promise((resolve, reject) => {
        const writer = fs.createWriteStream(imagePath);
        response.data.pipe(writer);
        writer.on('finish', resolve);
        writer.on('error', reject);
      });
      
      // Add the downloaded image path to the images array
      videoObj.images.push(imagePath);
    } catch (err) {
      console.error('Error downloading thumbnail:', err);
    }
  }

  async processVideo(mediabook, videoObj) {
    const videoId = Math.random().toString(36).substring(2, 15);
    const videoPath = `/tmp/${videoId}.mp4`;
    
    try {
      // Download video
      await this.downloadVideo(mediabook, videoPath);
      
      // Extract frames
      const frames = await this.extractFrames(videoPath, videoId);
      
      // Add frames to video object
      frames.forEach(frame => {
        videoObj.images.push(frame);
      });
      
      // Clean up video file
      fs.unlinkSync(videoPath);
    } catch (err) {
      console.error('Error processing video:', err);
    }
  }

  async downloadVideo(url, path) {
    const response = await axios({
      method: 'GET',
      url: url,
      responseType: 'stream'
    });
    
    return new Promise((resolve, reject) => {
      const writer = fs.createWriteStream(path);
      response.data.pipe(writer);
      writer.on('finish', resolve);
      writer.on('error', reject);
    });
  }

  async extractFrames(videoPath, videoId) {
    return new Promise((resolve, reject) => {
      const outputDir = `/tmp/${videoId}`;
      fs.mkdirSync(outputDir, { recursive: true });
      
      const ffmpeg = spawn('ffmpeg', [
        '-i', videoPath,
        '-vf', 'fps=1/1', // Extract 1 frame every 1 seconds
        '-vframes', '9',  // Limit to 9 frames
        `${outputDir}/frame_%03d.jpg`
      ]);
      
      ffmpeg.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`FFmpeg process exited with code ${code}`));
          return;
        }
        
        // Get the generated frame files
        const frames = fs.readdirSync(outputDir)
          .filter(file => file.startsWith('frame_'))
          .map(file => `${outputDir}/${file}`);
        
        resolve(frames);
      });
      
      ffmpeg.stderr.on('data', (data) => {
        console.log(`FFmpeg: ${data}`);
      });
    });
  }

  getVideos() {
    return this.videos;
  }
}

module.exports = fetchVideos;