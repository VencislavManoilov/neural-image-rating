const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');
const { predictImage, predictBatch } = require('./predictImage');
const logger = require('./logger');

const SITE_URL = process.env.SITE_URL;

class fetchVideos {
  constructor(html, labelForRating = null) {
    this.html = html;
    this.videos = [];
    this.labelForRating = labelForRating;
  }

  // Helper method for creating cross-platform temp paths
  getTempPath(filename) {
    return path.join(os.tmpdir(), filename);
  }

  async fetchVideos() {
    try {
      // Parse HTML with cheerio
      const $ = cheerio.load(this.html);
      
      // Log whether we're rating videos
      if (this.labelForRating) {
        logger.info(`Will rate videos using label: ${this.labelForRating}`);
      }
      
      // Create an array to hold all download promises
      const downloadPromises = [];
      
      // Limit to 5 videos for testing
      const maxVideos = 65;
      let videoCount = 0;
      
      // Find all elements with data-mediumthumb attribute
      $('[data-mediumthumb]').each((i, element) => {
        // Limit the number of videos for testing purposes
        if (videoCount >= maxVideos) return;
        videoCount++;
        
        // Get the parent anchor element to extract the href (video URL)
        const anchorElement = $(element).closest('a');
        const videoHref = anchorElement.attr('href');
        const videoUrl = videoHref ? `${SITE_URL}${videoHref}` : null;
        
        const thumbnailUrl = $(element).attr('data-mediumthumb');
        const mediabook = $(element).attr('data-mediabook');
        const newVideo = {
          images: [],
          ratings: [],
          averageRating: null,
          url: videoUrl // Add the video URL to the object
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
              // If a label is provided, predict ratings for all images
              if (this.labelForRating) {
                logger.info(`Rating video ${this.videos.length + 1} with ${newVideo.images.length} images`);
                await this.predictRatings(newVideo);
              }
              
              this.videos.push(newVideo);
            }
          } catch (err) {
            logger.error('Error processing media element:', err);
          }
        })();

        // Add this promise to our array of promises to track
        downloadPromises.push(processVideoPromise);
      });

      // Wait for all downloads to complete
      await Promise.all(downloadPromises);
      
      // Debug log the videos array to see if ratings were set
      if (this.labelForRating) {
        logger.info(`Number of videos with ratings: ${this.videos.filter(v => v.ratings && v.ratings.length > 0).length}`);
        // Log the first video's ratings as an example
        if (this.videos.length > 0) {
          logger.info(`First video ratings: ${JSON.stringify(this.videos[0].ratings)}`);
        }
      }
      
      // Sort videos by average rating (highest first)
      if (this.labelForRating) {
        this.videos.sort((a, b) => (b.averageRating || 0) - (a.averageRating || 0));
      }
      
      logger.info(`Videos fetched successfully: ${this.videos.length} videos${this.labelForRating ? ' with ratings' : ''}`);
      return this.videos;
    } catch (error) {
      logger.error('Error fetching videos:', error);
      throw error;
    }
  }

  async downloadThumbnail(url, videoObj) {
    try {
      const imageId = Math.random().toString(36).substring(2, 15);
      const imagePath = this.getTempPath(`${imageId}.jpg`);
      
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
      logger.error('Error downloading thumbnail:', err);
    }
  }

  async processVideo(mediabook, videoObj) {
    const videoId = Math.random().toString(36).substring(2, 15);
    const videoPath = this.getTempPath(`${videoId}.mp4`);
    
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
      logger.error('Error processing video:', err);
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
      const outputDir = this.getTempPath(videoId);
      fs.mkdirSync(outputDir, { recursive: true });
      
      try {
        const ffmpeg = spawn('ffmpeg', [
          '-i', videoPath,
          '-vf', 'fps=1/1', // Extract 1 frame every 1 seconds
          '-vframes', '9',  // Limit to 9 frames
          path.join(outputDir, 'frame_%03d.jpg')
        ]);
        
        ffmpeg.on('close', (code) => {
          if (code !== 0) {
            logger.error(`FFmpeg process exited with code ${code}`);
            return resolve([]); // Return empty array instead of rejecting
          }
          
          // Get the generated frame files
          const frames = fs.readdirSync(outputDir)
            .filter(file => file.startsWith('frame_'))
            .map(file => path.join(outputDir, file));
          
          resolve(frames);
        });
        
        ffmpeg.stderr.on('data', (data) => {
          logger.info(`FFmpeg: ${data.toString()}`);
        });
        
        // Handle the spawn error (which occurs when ffmpeg isn't found)
        ffmpeg.on('error', (err) => {
          logger.error(`Error spawning FFmpeg: ${err.message}`);
          logger.info('Please install FFmpeg: https://ffmpeg.org/download.html');
          resolve([]); // Return empty array to avoid breaking the chain
        });
      } catch (err) {
        logger.error(`Error in extractFrames: ${err.message}`);
        resolve([]); // Return empty array to avoid breaking the chain
      }
    });
  }

  async predictRatings(videoObj) {
    if (!this.labelForRating || videoObj.images.length === 0) {
      videoObj.averageRating = null;
      return;
    }
    
    logger.info(`Predicting ratings for ${videoObj.images.length} images using model ${this.labelForRating}`);
    
    try {
      // Use batch prediction for all images at once
      const batchResults = await predictBatch(videoObj.images, this.labelForRating);
      
      // Debug log the batch results
      logger.info(`Received batch predictions for ${batchResults.length} images`);
      
      // Create a mapping of imagePath to rating for easier access
      const ratingMap = {};
      batchResults.forEach(result => {
        ratingMap[result.imagePath] = result.rating;
      });
      
      // Initialize ratings array and fill with results in the same order as images
      videoObj.ratings = videoObj.images.map(imagePath => ratingMap[imagePath] || null);
      
      // Calculate average rating (excluding null values)
      const validRatings = videoObj.ratings.filter(rating => rating !== null);
      if (validRatings.length > 0) {
        const sum = validRatings.reduce((a, b) => a + b, 0);
        videoObj.averageRating = sum / validRatings.length;
        // Round to 2 decimal places for cleaner display
        videoObj.averageRating = Math.round(videoObj.averageRating * 100) / 100;
        
        // Debug log all ratings
        logger.info(`All ratings for video: ${JSON.stringify(videoObj.ratings)}`);
      } else {
        videoObj.averageRating = null;
      }
      
      logger.info(`Average rating for video: ${videoObj.averageRating} (${validRatings.length}/${videoObj.images.length} valid ratings)`);
    } catch (error) {
      logger.error(`Error predicting ratings for video: ${error.message}`);
      // Initialize empty ratings and null average if prediction fails
      videoObj.ratings = new Array(videoObj.images.length).fill(null);
      videoObj.averageRating = null;
    }
  }

  getVideos() {
    return this.videos;
  }
}

module.exports = fetchVideos;