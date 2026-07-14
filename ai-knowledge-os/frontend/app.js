const API_BASE = 'http://127.0.0.1:8000';

document.addEventListener('DOMContentLoaded', () => {
    // 1. Navigation Controller
    const tabButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const tabTitle = document.getElementById('tab-title');
    const tabDescription = document.getElementById('tab-description');

    const tabMetadata = {
        'ingest-tab': {
            title: 'Ingestion Hub',
            description: 'Upload files, scrape web content, and index them in your local vector database.'
        },
        'search-tab': {
            title: 'Semantic Search',
            description: 'Execute vector space searches to retrieve matched passages based on conceptual similarity.'
        },
        'chat-tab': {
            title: 'RAG Assistant',
            description: 'Ask questions about your uploaded documents and get answers grounded strictly in retrieved context.'
        }
    };

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // Switch tabs active classes
            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
            
            // Update workspace headers
            const meta = tabMetadata[targetTab];
            tabTitle.textContent = meta.title;
            tabDescription.textContent = meta.description;
        });
    });

    // 2. Drag & Drop File Selector Controller
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const selectedFileBadge = document.getElementById('selected-file');
    const fileNameText = document.getElementById('file-name-text');
    const removeFileBtn = document.getElementById('remove-file-btn');
    const fileUploadForm = document.getElementById('file-upload-form');

    // Click dropzone to open input
    dropzone.addEventListener('click', (e) => {
        if (e.target !== removeFileBtn && !removeFileBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', () => {
        handleFileSelection(fileInput.files[0]);
    });

    // Drag-over styling
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        handleFileSelection(file);
    });

    function handleFileSelection(file) {
        if (!file) return;
        fileNameText.textContent = file.name;
        selectedFileBadge.style.display = 'flex';
        // Keep files list
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
    }

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = '';
        selectedFileBadge.style.display = 'none';
    });

    // 3. File Ingestion Submit Controller
    fileUploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select a file to ingest first.');
            return;
        }

        const submitBtn = document.getElementById('file-submit-btn');
        const originalText = submitBtn.innerHTML;
        setButtonLoading(submitBtn, true);

        // Prep form data parameters
        const formData = new FormData();
        formData.append('file', file);
        formData.append('collection_name', document.getElementById('file-collection').value);
        formData.append('chunk_size', document.getElementById('file-chunk-size').value);
        formData.append('chunk_overlap', document.getElementById('file-chunk-overlap').value);
        formData.append('generate_summary', document.getElementById('file-summary').checked);

        // Add loading entry in logs
        const logId = addLogEntry(file.name, document.getElementById('file-collection').value, 'File');

        try {
            const response = await fetch(`${API_BASE}/api/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Ingestion failure');
            }

            const data = await response.json();
            updateLogEntry(logId, 'success', data.chunks_count, data.summary);
            // Clear inputs
            fileInput.value = '';
            selectedFileBadge.style.display = 'none';
        } catch (error) {
            console.error(error);
            updateLogEntry(logId, 'error', 0, error.message);
        } finally {
            setButtonLoading(submitBtn, false, originalText);
        }
    });

    // 4. URL Ingestion Submit Controller
    const urlIngestForm = document.getElementById('url-ingest-form');
    urlIngestForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const submitBtn = document.getElementById('url-submit-btn');
        const originalText = submitBtn.innerHTML;
        setButtonLoading(submitBtn, true);

        const url = document.getElementById('scrape-url').value;
        const collection = document.getElementById('url-collection').value;
        const chunk_size = parseInt(document.getElementById('url-chunk-size').value);
        const chunk_overlap = parseInt(document.getElementById('url-chunk-overlap').value);
        const generate_summary = document.getElementById('url-summary').checked;

        const logId = addLogEntry(url, collection, 'Link');

        try {
            const response = await fetch(`${API_BASE}/api/url`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    url,
                    collection_name: collection,
                    chunk_size,
                    chunk_overlap,
                    generate_summary
                })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Scraping failure');
            }

            const data = await response.json();
            updateLogEntry(logId, 'success', data.total_chunks, data.summary);
            urlIngestForm.reset();
        } catch (error) {
            console.error(error);
            updateLogEntry(logId, 'error', 0, error.message);
        } finally {
            setButtonLoading(submitBtn, false, originalText);
        }
    });

    // Log Table Utility Helper
    let logCounter = 0;
    function addLogEntry(source, collection, type) {
        logCounter++;
        const logId = `log-${logCounter}`;
        const tbody = document.getElementById('log-body');
        const emptyRow = tbody.querySelector('.empty-row');
        if (emptyRow) emptyRow.remove();

        const tr = document.createElement('tr');
        tr.id = logId;
        tr.innerHTML = `
            <td><span class="log-summary-cell" title="${source}">${source}</span></td>
            <td><code>${collection}</code></td>
            <td><i class="fa-solid ${type === 'File' ? 'fa-file-lines' : 'fa-link'}"></i> ${type}</td>
            <td>-</td>
            <td><span class="badge-status badge-loading"><i class="fa-solid fa-spinner fa-spin"></i> Processing</span></td>
        `;
        tbody.prepend(tr);
        return logId;
    }

    function updateLogEntry(id, status, chunks, summary) {
        const tr = document.getElementById(id);
        if (!tr) return;

        const cells = tr.querySelectorAll('td');
        cells[3].textContent = chunks;

        if (status === 'success') {
            const summaryText = summary ? `Summary: ${summary}` : 'Indexed successfully';
            cells[4].innerHTML = `
                <span class="badge-status badge-success"><i class="fa-solid fa-circle-check"></i> Success</span>
                <p style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 6px; max-width: 320px; line-height: 1.4;">${summaryText}</p>
            `;
        } else {
            cells[4].innerHTML = `
                <span class="badge-status badge-error"><i class="fa-solid fa-circle-exclamation"></i> Error</span>
                <p style="font-size: 0.8rem; color: var(--error-color); margin-top: 6px; max-width: 320px; line-height: 1.4;">${summary || 'Unknown failure'}</p>
            `;
        }
    }

    // 5. Semantic Search Submit Controller
    const searchForm = document.getElementById('search-form');
    const searchResultsContainer = document.getElementById('search-results');

    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = document.getElementById('search-query').value;
        const collection = document.getElementById('search-collection').value;
        const limit = parseInt(document.getElementById('search-limit').value);
        const filterKey = document.getElementById('search-filter-key').value.trim();
        const filterVal = document.getElementById('search-filter-val').value.trim();

        searchResultsContainer.innerHTML = `
            <div class="search-placeholder">
                <i class="fa-solid fa-circle-notch fa-spin" style="color: var(--accent-color); font-size: 2.5rem;"></i>
                <p>Querying semantic index...</p>
            </div>
        `;

        let filter_metadata = null;
        if (filterKey && filterVal) {
            filter_metadata = { [filterKey]: filterVal };
        }

        try {
            const response = await fetch(`${API_BASE}/api/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    collection_name: collection,
                    limit,
                    filter_metadata
                })
            });

            if (!response.ok) {
                throw new Error('Search failed');
            }

            const data = await response.json();
            renderSearchResults(data.results);
        } catch (error) {
            console.error(error);
            searchResultsContainer.innerHTML = `
                <div class="search-placeholder" style="color: var(--error-color);">
                    <i class="fa-solid fa-triangle-exclamation"></i>
                    <p>Search failed: ${error.message}</p>
                </div>
            `;
        }
    });

    function renderSearchResults(results) {
        if (!results || results.length === 0) {
            searchResultsContainer.innerHTML = `
                <div class="search-placeholder">
                    <i class="fa-solid fa-magnifying-glass-minus"></i>
                    <p>No conceptual matches found in this collection.</p>
                </div>
            `;
            return;
        }

        searchResultsContainer.innerHTML = results.map(res => {
            const scorePercent = Math.round(res.score * 100);
            
            // Format metadata badges
            let metaHtml = '';
            for (const [key, val] of Object.entries(res.metadata)) {
                metaHtml += `<span><strong>${key}:</strong> ${val}</span>`;
            }

            return `
                <div class="glass-card result-card">
                    <div class="result-card-header">
                        <span class="result-source"><i class="fa-solid fa-file-invoice"></i> ${res.source}</span>
                        <span class="result-score">${scorePercent}% Match</span>
                    </div>
                    <p class="result-text">${escapeHtml(res.text)}</p>
                    <div class="result-meta">
                        ${metaHtml}
                    </div>
                </div>
            `;
        }).join('');
    }

    // 6. RAG Assistant Chat Controller
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const citationsList = document.getElementById('citations-list');

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = chatInput.value.trim();
        if (!question) return;

        // 1. Add user bubble
        appendMessage('user', question);
        chatInput.value = '';

        // 2. Add loading bot bubble
        const botMsgId = appendMessage('assistant', '<i class="fa-solid fa-circle-notch fa-spin"></i> Processing contextual answer...');
        
        // Clear citations placeholder
        citationsList.innerHTML = `
            <div class="citation-placeholder">
                <i class="fa-solid fa-circle-notch fa-spin" style="color: var(--secondary-accent);"></i>
                <p>Matching citations...</p>
            </div>
        `;

        try {
            const response = await fetch(`${API_BASE}/api/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Generation failed');
            }

            const data = await response.json();
            
            // Update bot bubble with answers
            updateMessage(botMsgId, data.answer);
            
            // Render citations side drawer
            renderCitations(data.citations);
        } catch (error) {
            console.error(error);
            updateMessage(botMsgId, `Failed to retrieve RAG response: ${error.message}`, true);
            citationsList.innerHTML = `
                <div class="citation-placeholder" style="color: var(--error-color);">
                    <i class="fa-solid fa-triangle-exclamation"></i>
                    <p>Citations loading failed.</p>
                </div>
            `;
        }
    });

    let msgIdCounter = 0;
    function appendMessage(role, content) {
        msgIdCounter++;
        const id = `msg-${msgIdCounter}`;
        const div = document.createElement('div');
        div.className = `message ${role}`;
        div.id = id;
        div.innerHTML = `
            <div class="message-bubble">
                <p>${content}</p>
            </div>
        `;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return id;
    }

    function updateMessage(id, text, isError = false) {
        const msgDiv = document.getElementById(id);
        if (!msgDiv) return;
        const bubble = msgDiv.querySelector('.message-bubble p');
        bubble.innerHTML = isError ? `<span style="color: var(--error-color);">${text}</span>` : formatTextMarkdown(text);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function renderCitations(citations) {
        if (!citations || citations.length === 0) {
            citationsList.innerHTML = `
                <div class="citation-placeholder">
                    <i class="fa-solid fa-circle-question"></i>
                    <p>No citations found for this answer.</p>
                </div>
            `;
            return;
        }

        citationsList.innerHTML = citations.map((cit, index) => `
            <div class="citation-item">
                <div class="citation-header">
                    <span class="citation-index">[Reference ${index + 1}]</span>
                    <span class="citation-score">Score: ${cit.score}</span>
                </div>
                <div class="citation-source" title="${cit.source}">
                    <i class="fa-solid fa-file-lines"></i> ${cit.source}
                </div>
            </div>
        `).join('');
    }

    // Helper formatting tools
    function setButtonLoading(btn, isLoading, originalText = '') {
        if (isLoading) {
            btn.disabled = true;
            btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> <span>Processing...</span>`;
        } else {
            btn.disabled = false;
            btn.innerHTML = originalText;
        }
    }

    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function formatTextMarkdown(text) {
        // Simple regex to format citations [1], [2] to bold colored markers
        let formatted = escapeHtml(text);
        formatted = formatted.replace(/\[(\d+)\]/g, '<strong style="color: var(--secondary-accent); font-weight: 700;">[$1]</strong>');
        // Simple linebreaks
        return formatted.replace(/\n/g, '<br>');
    }
});
