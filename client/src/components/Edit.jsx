import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import Rate from "./Rate";

const URL = process.env.REACT_APP_API_URL || "http://localhost:8080";

const Edit = () => {
    const [label, setLabel] = useState(null);
    
    const { id } = useParams();

    useEffect(() => {
        const fetchLabel = async () => {
            if(!id) {
                return;
            }

            try {
                const response = await fetch(`${URL}/labels/get/${id}`, {
                    method: "GET",
                    headers: {
                        Authorization: `Bearer ${localStorage.getItem("token")}`,
                        "Content-Type": "application/json",
                    },
                });

                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }

                const data = await response.json();
                setLabel({
                    name: data.labelsDetails.label,
                    labels: data.labels,
                });
            } catch (error) {
                console.error("Error fetching label:", error);
            }
        };

        fetchLabel();
    }, [id]);

    return (
        <div className="edit-container">
            <h2>Edit Label</h2>
            <Rate label={label} />
        </div>
    );
};

export default Edit;