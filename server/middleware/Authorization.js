const jwt = require('jsonwebtoken');
const knex = require('../knex');
const logger = require('../utils/logger');

// JWT secret key - should match the one in auth.js
const JWT_SECRET = process.env.JWT_SECRET || 'your-jwt-secret-key';

/**
 * Authorization middleware to validate JWT tokens
 * and attach user to request object
 */
async function Authorization(req, res, next) {
  try {
    // Get token from header
    const authHeader = req.header('Authorization');
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ message: 'Authorization token required' });
    }

    // Extract token without "Bearer " prefix
    const token = authHeader.substring(7);
    
    // Verify token
    const decoded = jwt.verify(token, JWT_SECRET);
    
    // Find user by email from token
    const user = await knex('users')
      .where({ email: decoded.email })
      .select('id', 'username', 'email', 'role')
      .first();
      
    if (!user) {
      return res.status(401).json({ message: 'Invalid token - user not found' });
    }
    
    // Attach user to request object
    req.user = user;
    
    // Continue to next middleware
    next();
    
  } catch (error) {
    logger.error('Authorization error:' + error);
    if (error.name === 'JsonWebTokenError' || error.name === 'TokenExpiredError') {
      return res.status(401).json({ message: 'Invalid or expired token' });
    }
    res.status(500).json({ message: 'Server error during authorization' });
  }
}

module.exports = Authorization;
