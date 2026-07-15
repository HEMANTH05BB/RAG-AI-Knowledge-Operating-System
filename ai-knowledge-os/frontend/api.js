const API_BASE = window.location.origin.includes('127.0.0.1') || window.location.origin.includes('localhost') ? window.location.origin : 'http://127.0.0.1:8000';

class ApiClient {
    /**
     * Helper to process the fetch response, handle errors and parse JSON.
     */
    static async handleResponse(response) {
        if (!response.ok) {
            let errorMsg = 'An error occurred';
            try {
                const errorData = await response.json();
                errorMsg = errorData.detail || errorData.message || JSON.stringify(errorData);
            } catch (e) {
                errorMsg = response.statusText || errorMsg;
            }
            throw new Error(errorMsg);
        }
        return response.json();
    }

    /**
     * Upload a document via Form Data
     */
    static async uploadDocument(formData) {
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });
        return this.handleResponse(response);
    }

    /**
     * Ingest a webpage or YouTube URL
     */
    static async scrapeUrl(payload) {
        const response = await fetch(`${API_BASE}/api/url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        return this.handleResponse(response);
    }

    /**
     * Query semantic matching database vectors
     */
    static async semanticSearch(payload) {
        const response = await fetch(`${API_BASE}/api/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        return this.handleResponse(response);
    }

    /**
     * Send Q&A request to RAG endpoint
     */
    static async askQuestion(payload) {
        const response = await fetch(`${API_BASE}/api/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        return this.handleResponse(response);
    }

    /**
     * Summarize document chunks
     */
    static async summarizeDocument(payload) {
        const response = await fetch(`${API_BASE}/api/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        return this.handleResponse(response);
    }

    /**
     * List all unique document sources
     */
    static async listDocuments(collectionName = 'knowledge_base') {
        const response = await fetch(`${API_BASE}/api/upload/list?collection_name=${encodeURIComponent(collectionName)}`);
        return this.handleResponse(response);
    }

    /**
     * Check API Health
     */
    static async checkHealth() {
        const response = await fetch(`${API_BASE}/health`);
        return this.handleResponse(response);
    }
}
