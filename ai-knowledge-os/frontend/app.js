const API_BASE = window.location.origin.includes('127.0.0.1') || window.location.origin.includes('localhost') ? window.location.origin : 'http://127.0.0.1:8000';

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
            loadIndexedDocuments();
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
            loadIndexedDocuments();
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
            
            // Update bot bubble with answers and inline citations
            updateMessage(botMsgId, data.answer, false, data.citations);
            
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

    function updateMessage(id, text, isError = false, citations = null) {
        const msgDiv = document.getElementById(id);
        if (!msgDiv) return;
        const bubble = msgDiv.querySelector('.message-bubble p');
        bubble.innerHTML = isError ? `<span style="color: var(--error-color);">${text}</span>` : formatTextMarkdown(text);
        
        // Render citations display directly under the answer text
        if (citations && citations.length > 0) {
            const citationsDiv = document.createElement('div');
            citationsDiv.className = 'message-citations';
            citationsDiv.style.marginTop = '12px';
            citationsDiv.style.paddingTop = '10px';
            citationsDiv.style.borderTop = '1px solid rgba(255, 255, 255, 0.08)';
            citationsDiv.style.fontSize = '0.8rem';
            citationsDiv.style.color = 'var(--text-secondary)';
            
            let citationsHtml = '<strong style="display: block; margin-bottom: 6px; color: var(--text-primary);"><i class="fa-solid fa-quote-left" style="font-size: 0.75rem; margin-right: 4px;"></i> Sources used:</strong>';
            citationsHtml += '<div style="display: flex; flex-direction: column; gap: 4px;">';
            citations.forEach((cit, index) => {
                const scorePercent = Math.round(cit.score * 100);
                const docName = cit.source.split('/').pop();
                citationsHtml += `
                    <span style="display: flex; align-items: center; gap: 6px;">
                        <strong style="color: var(--secondary-accent);">[${index + 1}]</strong> 
                        <span title="${cit.source}" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 75%;">${docName}</span> 
                        <span style="color: var(--success-color); font-weight: 600;">(${scorePercent}% match)</span>
                    </span>
                `;
            });
            citationsHtml += '</div>';
            citationsDiv.innerHTML = citationsHtml;
            msgDiv.querySelector('.message-bubble').appendChild(citationsDiv);
        }
        
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

    // 8. Document List Controller
    const documentList = document.getElementById('document-list');
    const refreshDocsBtn = document.getElementById('refresh-docs-btn');

    async function loadIndexedDocuments() {
        try {
            const response = await fetch(`${API_BASE}/api/upload/list`);
            if (!response.ok) throw new Error('Failed to load documents');
            
            const data = await response.json();
            renderDocumentList(data.documents);
        } catch (error) {
            console.error('Error loading documents:', error);
            documentList.innerHTML = `
                <li style="color: var(--error-color); text-align: center; padding: 1rem 0;">
                    <i class="fa-solid fa-triangle-exclamation" style="display: block; margin-bottom: 8px;"></i>
                    Failed to fetch document list.
                </li>
            `;
        }
    }

    function renderDocumentList(documents) {
        if (!documents || documents.length === 0) {
            documentList.innerHTML = `
                <li class="empty-doc-row" style="color: var(--text-muted); text-align: center; padding: 2rem 0; width: 100%;">
                    <i class="fa-solid fa-box-open" style="font-size: 2rem; margin-bottom: 10px; display: block; color: var(--text-muted);"></i>
                    No indexed documents found in database.
                </li>
            `;
            return;
        }

        documentList.innerHTML = documents.map(doc => {
            // Determine icon by type
            let icon = 'fa-file-lines';
            if (doc.endsWith('.pdf')) icon = 'fa-file-pdf';
            else if (doc.endsWith('.docx') || doc.endsWith('.doc')) icon = 'fa-file-word';
            else if (doc.endsWith('.pptx') || doc.endsWith('.ppt')) icon = 'fa-file-powerpoint';
            else if (doc.endsWith('.html') || doc.endsWith('.htm')) icon = 'fa-file-code';
            else if (doc.startsWith('http://') || doc.startsWith('https://')) icon = 'fa-link';

            const displayName = doc.startsWith('YouTube (ID:') ? doc : doc.split('/').pop();

            return `
                <li class="document-item" style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border-color); border-radius: 8px; transition: background 0.2s;">
                    <div style="display: flex; align-items: center; gap: 10px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; max-width: 70%;">
                        <i class="fa-solid ${icon}" style="color: var(--accent-color); font-size: 1.1rem;"></i>
                        <span class="doc-name" style="font-weight: 500; font-size: 0.9rem;" title="${doc}">${displayName}</span>
                    </div>
                    <button class="btn btn-secondary summarize-doc-btn" data-filename="${doc}" style="padding: 4px 10px; font-size: 0.8rem; width: auto; height: auto; min-width: unset; margin: 0; display: flex; align-items: center; gap: 6px;">
                        <i class="fa-solid fa-wand-magic-sparkles"></i> Summarize
                    </button>
                </li>
            `;
        }).join('');

        // Bind clicks to newly rendered buttons
        documentList.querySelectorAll('.summarize-doc-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const filename = btn.getAttribute('data-filename');
                handleDocumentSummarize(filename, btn);
            });
        });
    }

    refreshDocsBtn.addEventListener('click', loadIndexedDocuments);

    // 9. Document Summarize Controller
    const summaryModal = document.getElementById('summary-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    const closeModalBtn = document.getElementById('close-modal-btn');

    async function handleDocumentSummarize(filename, btn) {
        const originalText = btn.innerHTML;
        setButtonLoading(btn, true);
        
        // Open modal with loading state
        modalTitle.textContent = `Summarizing: ${filename.split('/').pop()}`;
        modalBody.innerHTML = `
            <div style="text-align: center; padding: 3rem 0; color: var(--text-secondary);">
                <i class="fa-solid fa-circle-notch fa-spin" style="font-size: 2.5rem; color: var(--accent-color); margin-bottom: 12px;"></i>
                <p>Retrieving vector chunks and generating AI summary using Llama 3.3 70B...</p>
            </div>
        `;
        summaryModal.style.display = 'flex';

        try {
            const response = await fetch(`${API_BASE}/api/summarize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Summarization failed');
            }

            const data = await response.json();
            modalBody.innerHTML = formatSummaryMarkdown(data.summary);
        } catch (error) {
            console.error('Error generating summary:', error);
            modalBody.innerHTML = `
                <div style="text-align: center; padding: 2rem 0; color: var(--error-color);">
                    <i class="fa-solid fa-triangle-exclamation" style="font-size: 2.5rem; margin-bottom: 12px;"></i>
                    <p>Failed to generate summary: ${error.message}</p>
                </div>
            `;
        } finally {
            setButtonLoading(btn, false, originalText);
        }
    }

    // Close Modal Bindings
    closeModalBtn.addEventListener('click', () => {
        summaryModal.style.display = 'none';
    });

    window.addEventListener('click', (e) => {
        if (e.target === summaryModal) {
            summaryModal.style.display = 'none';
        }
    });

    function formatSummaryMarkdown(text) {
        let html = escapeHtml(text);
        
        // Convert Markdown bold (**text**) to HTML <strong>
        html = html.replace(/\*\*(.*?)\*\?/g, '<strong>$1</strong>');
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Convert Bullet points (- text or * text) to lists
        html = html.replace(/^\s*[-*]\s+(.*?)$/gm, '<li style="margin-left: 20px; margin-bottom: 6px;">$1</li>');
        
        // Convert numbered lists
        html = html.replace(/^\s*(\d+)\.\s+(.*?)$/gm, '<div style="margin-left: 20px; margin-bottom: 6px;"><strong>$1.</strong> $2</div>');
        
        // Linebreaks
        html = html.replace(/\n/g, '<br>');
        
        return html;
    }

    // 10. API Health Check Controller
    async function checkApiHealth() {
        const dot = document.querySelector('.pulse-dot');
        const text = dot.nextElementSibling;
        try {
            const res = await fetch(`${API_BASE}/health`);
            if (res.ok) {
                dot.style.background = 'var(--success-color)';
                dot.style.boxShadow = '0 0 0 0 rgba(16, 185, 129, 0.7)';
                text.textContent = 'API Status: Online';
            } else {
                throw new Error();
            }
        } catch {
            dot.style.background = 'var(--error-color)';
            dot.style.animation = 'none';
            text.textContent = 'API Status: Offline';
            text.style.color = 'var(--error-color)';
        }
    }

    // Initial page load triggers
    checkApiHealth();
    loadIndexedDocuments();
});
