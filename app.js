// DeficiVision AI 2.0 - SPA Client Logic

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewBox = document.getElementById('preview-box');
    const previewImg = document.getElementById('preview-img');
    const btnAnalyze = document.getElementById('btn-analyze');
    const placeholderResults = document.getElementById('results-placeholder');
    const contentResults = document.getElementById('results-content');
    
    let currentImageBase64 = null;

    // Deficiency Knowledge Catalog
    const LIBRARY_DATA = {
        'Vitamin A': {
            symptoms: 'Night blindness, dry eyes (xerophthalmia), Bitot spots, dermal hyperkeratosis.',
            foods: ['🥕 Carrots', '🍠 Sweet Potatoes', '🥬 Spinach', '🥚 Eggs'],
            care: ['Increase beta-carotene rich dark leafy greens.', 'Consult an ophthalmologist for eye checkup.']
        },
        'Vitamin B2': {
            symptoms: 'Angular cheilitis (cracked lips), magenta tongue, glossitis, red/itchy eyes.',
            foods: ['🥛 Dairy Milk', '🥚 Eggs', '🌰 Almonds', '🍄 Mushrooms'],
            care: ['Consume riboflavin-rich dairy and whole grains.', 'Take Vitamin B-complex daily.']
        },
        'Vitamin B3': {
            symptoms: 'Pellagra (dermatitis, red lesions, sensitive skin, tongue swelling).',
            foods: ['🍗 Poultry', '🐟 Tuna & Salmon', '🍚 Brown Rice', '🥜 Peanuts'],
            care: ['Eat lean meats, fish, and legumes.', 'Avoid excessive UV sun exposure on rash areas.']
        },
        'Vitamin B7': {
            symptoms: 'Hair thinning/alopecia, red scaly rash around facial orifices, brittle nails.',
            foods: ['🥚 Egg Yolks', '🥜 Nuts', '🌻 Seeds', '🍠 Sweet Potatoes'],
            care: ['Eat biotin-rich whole foods.', 'Take 30-100 mcg daily biotin supplement.']
        },
        'Vitamin B12': {
            symptoms: 'Pale skin, smooth inflamed tongue (glossitis), fatigue, mouth ulcers.',
            foods: ['🥩 Lean Beef', '🐟 Salmon', '🥛 Fortified Dairy', '🥣 Fortified Cereal'],
            care: ['Incorporate B12-enriched animal proteins or fortified plant milk.', 'Check serum B12 levels.']
        },
        'Vitamin C': {
            symptoms: 'Bleeding gums (scurvy), petechiae bruising, slow wound healing, dry skin.',
            foods: ['🍊 Citrus Fruits', '🫑 Bell Peppers', '🥝 Kiwi Fruit', '🍓 Strawberries'],
            care: ['Increase fresh citrus and raw capsicum intake.', 'Take daily Ascorbic Acid (250mg-500mg).']
        }
    };

    // Render Knowledge Library
    const libGrid = document.getElementById('library-grid');
    if (libGrid) {
        libGrid.innerHTML = Object.entries(LIBRARY_DATA).map(([name, item]) => `
            <div class="glass-panel lib-card">
                <div class="lib-title">👁️ ${name}</div>
                <div class="lib-symptoms"><strong>Indicators:</strong> ${item.symptoms}</div>
                <div>${item.foods.map(f => `<span class="sample-chip">${f}</span>`).join(' ')}</div>
            </div>
        `).join('');
    }

    // Drag & Drop Handlers
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPG, PNG, WebP).');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            currentImageBase64 = e.target.result;
            previewImg.src = currentImageBase64;
            previewBox.classList.remove('hidden');
            btnAnalyze.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Quick Sample Picker
    window.loadSample = function(type) {
        // Create canvas placeholder sample
        const canvas = document.createElement('canvas');
        canvas.width = 300;
        canvas.height = 300;
        const ctx = canvas.getContext('2d');
        
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, 300, 300);
        
        ctx.beginPath();
        ctx.arc(150, 150, 80, 0, Math.PI * 2);
        ctx.fillStyle = type.includes('Eye') ? '#10b981' : type.includes('Lip') ? '#f43f5e' : '#f59e0b';
        ctx.fill();
        
        currentImageBase64 = canvas.toDataURL('image/jpeg');
        previewImg.src = currentImageBase64;
        previewBox.classList.remove('hidden');
        btnAnalyze.disabled = false;
    };

    // Analyze Click Handler
    btnAnalyze.addEventListener('click', async () => {
        if (!currentImageBase64) return;
        
        btnAnalyze.disabled = true;
        btnAnalyze.innerHTML = '⏳ Running ResNet50 + ViT Pipeline...';

        try {
            const resp = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: currentImageBase64 })
            });

            const res = await resp.json();
            
            if (res.success) {
                renderResults(res.data);
            } else {
                alert('Analysis failed: ' + res.error);
            }
        } catch (err) {
            console.error(err);
            alert('Network error while connecting to PyTorch backend server.');
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.innerHTML = '✨ Run Dual-Engine AI Scan';
        }
    });

    function renderResults(data) {
        placeholderResults.classList.add('hidden');
        contentResults.classList.remove('hidden');
        
        document.getElementById('res-class').textContent = data.predicted_class;
        document.getElementById('res-conf').textContent = `${data.confidence}%`;
        document.getElementById('res-level').textContent = data.confidence_level;
        
        // Render Gauge SVG ring
        const fill = document.getElementById('gauge-fill');
        const circumference = 226; // 2 * pi * 36
        const offset = circumference - (data.confidence / 100 * circumference);
        fill.style.strokeDasharray = `${circumference}`;
        fill.style.strokeDashoffset = `${offset}`;

        // Render Probabilities
        const probList = document.getElementById('prob-bars-list');
        probList.innerHTML = Object.entries(data.probabilities).map(([cls, pct]) => `
            <div class="prob-row">
                <div class="prob-name">${cls}</div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: ${pct}%"></div>
                </div>
                <div class="prob-pct">${pct}%</div>
            </div>
        `).join('');

        // Render Recommendations
        const rec = LIBRARY_DATA[data.predicted_class] || LIBRARY_DATA['Vitamin A'];
        document.getElementById('diet-grid').innerHTML = rec.foods.map(f => `<div class="diet-card">${f}</div>`).join('');
        document.getElementById('care-steps-list').innerHTML = rec.care.map(c => `<li>${c}</li>`).join('');
    }

    // Modal consultation triggers
    const modal = document.getElementById('consult-modal');
    document.getElementById('btn-consult')?.addEventListener('click', () => modal.classList.remove('hidden'));
    document.getElementById('btn-close-modal')?.addEventListener('click', () => modal.classList.add('hidden'));
});
