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

  // Clean up temporary files created for a video
  cleanupTempFiles(videoObj) {
    try {
      // Remove all image files
      if (videoObj.images && videoObj.images.length > 0) {
        videoObj.images.forEach(imgPath => {
          if (fs.existsSync(imgPath)) {
            fs.unlinkSync(imgPath);
          }
        });
      }
      
      // Check if any images are in a temp directory and remove the directory too
      const tempDirs = new Set();
      videoObj.images.forEach(imgPath => {
        const dir = path.dirname(imgPath);
        if (dir.includes(os.tmpdir())) {
          tempDirs.add(dir);
        }
      });
      
      // Remove empty temp directories
      tempDirs.forEach(dir => {
        if (fs.existsSync(dir)) {
          // Check if directory is empty
          if (fs.readdirSync(dir).length === 0) {
            fs.rmdirSync(dir);
          }
        }
      });
    } catch (err) {
      logger.error('Error cleaning up temp files:', err);
    }
  }

  async processVideoElement(element) {
    const { thumbnailUrl, mediabook, videoUrl } = element;
    
    const newVideo = {
      images: [],
      ratings: [],
      averageRating: null,
      url: videoUrl
    };
    
    try {
      // Download thumbnail if available
      if (thumbnailUrl && thumbnailUrl.startsWith('http')) {
        await this.downloadThumbnail(thumbnailUrl, newVideo);
      }

      // Download and process video if available
      if (mediabook && mediabook.startsWith('http')) {
        await this.processVideo(mediabook, newVideo);
      }

      // Only process videos with images
      if (newVideo.images.length > 0) {
        // If a label is provided, predict ratings for all images
        if (this.labelForRating) {
          logger.info(`Rating video with ${newVideo.images.length} images`);
          await this.predictRatings(newVideo);
        }
        
        // Create a copy to return without image paths
        const videoWithoutImages = {
          url: newVideo.url,
          ratings: [...newVideo.ratings],
          averageRating: newVideo.averageRating,
          // Keep a count of images but not the actual paths
          imageCount: newVideo.images.length
        };
        
        // Clean up temp files immediately after prediction to save memory
        this.cleanupTempFiles(newVideo);
        
        return videoWithoutImages;
      }
      
      return null;
    } catch (err) {
      logger.error('Error processing media element:', err);
      // Clean up any resources that might have been created
      this.cleanupTempFiles(newVideo);
      return null;
    }
  }

  async predictRatings(videoObj) {
    if (!this.labelForRating || videoObj.images.length === 0) {
      videoObj.averageRating = null;
      return;
    }
    
    logger.info(`Predicting ratings for ${videoObj.images.length} images using model ${this.labelForRating}`);
    
    try {
      // Break down image prediction into smaller batches
      const PREDICTION_BATCH_SIZE = 10;
      const results = [];
      
      for (let i = 0; i < videoObj.images.length; i += PREDICTION_BATCH_SIZE) {
        const imageBatch = videoObj.images.slice(i, i + PREDICTION_BATCH_SIZE);
        
        // Use batch prediction for current batch of images
        const batchResults = await predictBatch(imageBatch, this.labelForRating);
        
        if (batchResults && batchResults.length > 0) {
          results.push(...batchResults);
        }
        
        // Small delay to allow for garbage collection
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      // Create a mapping of imagePath to rating for easier access
      const ratingMap = {};
      results.forEach(result => {
        if (result && result.imagePath) {
          ratingMap[result.imagePath] = result.rating;
        }
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

  async fetchVideos() {
    if (!this.html) {
      return logger.error('HTML content is not provided.');
    }

    try {
      // Parse HTML with cheerio
      const $ = cheerio.load(this.html);
      
      // Log whether we're rating videos
      if (this.labelForRating) {
        logger.info(`Will rate videos using label: ${this.labelForRating}`);
      }
      
      // Limit to 65 videos as in original code
      const maxVideos = 65;
      let videoCount = 0;
      
      // Find all video elements
      const videoElements = [];
      $('[data-mediumthumb]').each((i, element) => {
        if (videoCount >= maxVideos) return;
        videoCount++;
        
        // Get the parent anchor element to extract the href (video URL)
        const anchorElement = $(element).closest('a');
        const videoHref = anchorElement.attr('href');
        const videoUrl = videoHref ? `${SITE_URL}${videoHref}` : null;
        
        const thumbnailUrl = $(element).attr('data-mediumthumb');
        const mediabook = $(element).attr('data-mediabook');
        
        videoElements.push({
          thumbnailUrl,
          mediabook,
          videoUrl
        });
      });
      
      logger.info(`Found ${videoElements.length} video elements to process`);
      
      // Process videos in smaller batches to control memory usage
      const BATCH_SIZE = 5;  // Process 5 videos at a time
      
      for (let i = 0; i < videoElements.length; i += BATCH_SIZE) {
        logger.info(`Processing batch ${Math.floor(i/BATCH_SIZE) + 1} of ${Math.ceil(videoElements.length/BATCH_SIZE)}`);
        
        const batch = videoElements.slice(i, i + BATCH_SIZE);
        const batchPromises = batch.map(element => this.processVideoElement(element));
        
        try {
          // Wait for current batch to finish before proceeding
          const processedVideos = await Promise.all(batchPromises);
          
          // Add only valid videos with ratings
          const validVideos = processedVideos.filter(video => video !== null);
          
          logger.info(`Batch completed with ${validVideos.length} valid videos of ${batch.length} processed`);
          
          this.videos.push(...validVideos);
        } catch (err) {
          logger.error(`Error processing batch: ${err.message}`);
          // Continue with next batch
        }
        
        // Force garbage collection between batches
        global.gc && global.gc();
      }
      
      // Sort videos by average rating (highest first) if we have ratings
      if (this.labelForRating) {
        this.videos.sort((a, b) => (b.averageRating || 0) - (a.averageRating || 0));
        
        // Log some debug info about ratings
        const videosWithRatings = this.videos.filter(v => v.averageRating !== null);
        logger.info(`Total videos with ratings: ${videosWithRatings.length} out of ${this.videos.length}`);
      }
      
      logger.info(`Videos fetched successfully: ${this.videos.length} videos${this.labelForRating ? ' with ratings' : ''}`);
      return this.videos;
    } catch (error) {
      logger.error('Error fetching videos:', error);
      return [];  // Return empty array instead of throwing to avoid crashing
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
      
      // Clean up video file immediately to save space
      if (fs.existsSync(videoPath)) {
        fs.unlinkSync(videoPath);
      }
    } catch (err) {
      logger.error('Error processing video:', err);
      // Make sure to clean up even on error
      if (fs.existsSync(videoPath)) {
        fs.unlinkSync(videoPath);
      }
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

  getVideos() {
    return this.videos;
  }
}

module.exports = fetchVideos;