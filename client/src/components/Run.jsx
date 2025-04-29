import { useState } from 'react'
import { useParams } from 'react-router-dom'
import axios from 'axios'
import './run.css'

const URL = process.env.REACT_APP_API_URL || 'http://localhost:8080'

const Run = () => {
	const labelName = useParams().label;
	const [loading, setLoading] = useState(false);

	const fetchRun = async () => {
		try {
			setLoading(true);

			const response = await axios.get(`${URL}/fetch-videos?label=${labelName}`, {
				headers: {
					Authorization: `Bearer ${localStorage.getItem('token')}`,
					'Content-Type': 'application/json'
				}
			});

			const data = await response.json();
			console.log('Run response:', data);
		} catch (error) {
			console.error('Error fetching run:', error);
		} finally {
			setLoading(false);
		}
	};
	
  return (
    <div className="run-container">
			{loading ? (
				<div className="loading">
					<div className="spinner"></div>
					<h2>Loading...</h2>
				</div>
			) : (
				<button className='button button-purple' onClick={fetchRun}>
					Run
				</button>
			)}
    </div>
  )
}

export default Run
