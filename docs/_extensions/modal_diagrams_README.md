# Modal Diagrams and Images System

This system provides clickable, expandable modal functionality for diagrams and images in the NeMo Automodel documentation.

## Features

- **Click to Expand**: Click any diagram or image to open in full-screen modal
- **Multiple Close Options**: 
  - Click the X button
  - Click outside the modal
  - Press Escape key
- **Visual Feedback**: Hover effects show clickable elements
- **Scroll Support**: Modal content has max-height: 80vh with overflow scrolling
- **Responsive Design**: Adapts to different screen sizes

## Implementation

### For Images

```html
<!-- HTML wrapper for clickable container -->
```{raw} html
<div id="unique-image-modal-container" style="cursor: pointer; display: inline-block; border: 2px solid transparent; border-radius: 8px; transition: border-color 0.3s ease;" 
     onmouseover="this.style.borderColor='#007acc'" 
     onmouseout="this.style.borderColor='transparent'"
     onclick="openImageModal('unique-image-modal', 'path/to/image.jpg', 'Image Caption')">
```

<!-- Your existing {image} directive -->
```{image} path/to/image.jpg
:alt: Image description
:class: bg-primary
:width: 400px
:align: center
```

<!-- Modal HTML -->
```{raw} html
</div>

<div id="unique-image-modal" style="display: none; position: fixed; z-index: 10000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.8); backdrop-filter: blur(3px);" onclick="closeImageModal('unique-image-modal')">
    <div style="position: relative; margin: auto; padding: 20px; width: 90%; max-width: 1200px; top: 50%; transform: translateY(-50%);">
        <span onclick="closeImageModal('unique-image-modal')" style="color: white; float: right; font-size: 28px; font-weight: bold; cursor: pointer; background: rgba(0,0,0,0.5); padding: 5px 10px; border-radius: 50%;">&times;</span>
        <div style="text-align: center; max-height: 80vh; overflow-y: auto;">
            <img id="unique-image-modal-img" src="" alt="" style="max-width: 100%; height: auto; border-radius: 8px;">
            <p id="unique-image-modal-caption" style="color: white; margin-top: 15px; font-size: 16px;"></p>
        </div>
    </div>
</div>

<script>
function openImageModal(modalId, imgSrc, caption) {
    const modal = document.getElementById(modalId);
    const modalImg = document.getElementById(modalId + '-img');
    const modalCaption = document.getElementById(modalId + '-caption');
    
    modal.style.display = 'block';
    modalImg.src = imgSrc;
    modalCaption.textContent = caption;
    
    document.body.style.overflow = 'hidden';
}

function closeImageModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

// Close modal with Escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const modals = document.querySelectorAll('[id*="-modal"]');
        modals.forEach(modal => {
            if (modal.style.display === 'block') {
                modal.style.display = 'none';
                document.body.style.overflow = 'auto';
            }
        });
    }
});
</script>
```

### For Mermaid Diagrams

```html
<!-- HTML wrapper for clickable container -->
```{raw} html
<div id="unique-diagram-modal-container" style="cursor: pointer; border: 2px solid transparent; border-radius: 8px; transition: border-color 0.3s ease; padding: 10px;" 
     onmouseover="this.style.borderColor='#007acc'; this.style.backgroundColor='rgba(0,122,204,0.05)'" 
     onmouseout="this.style.borderColor='transparent'; this.style.backgroundColor='transparent'"
     onclick="openDiagramModal('unique-diagram-modal', 'Diagram Title')">
```

<!-- Your existing Mermaid diagram -->
```{mermaid}
graph TD
    A[Node A] --> B[Node B]
    B --> C[Node C]
```

<!-- Modal HTML with enlarged diagram -->
```{raw} html
</div>

<div id="unique-diagram-modal" style="display: none; position: fixed; z-index: 10000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.8); backdrop-filter: blur(3px);" onclick="closeDiagramModal('unique-diagram-modal')">
    <div style="position: relative; margin: auto; padding: 20px; width: 95%; max-width: 1400px; top: 50%; transform: translateY(-50%);">
        <span onclick="closeDiagramModal('unique-diagram-modal')" style="color: white; float: right; font-size: 28px; font-weight: bold; cursor: pointer; background: rgba(0,0,0,0.5); padding: 5px 10px; border-radius: 50%;">&times;</span>
        <div style="text-align: center; max-height: 80vh; overflow-y: auto; background: white; border-radius: 12px; padding: 20px;">
            <h3 id="unique-diagram-modal-title" style="margin-top: 0; color: #333;"></h3>
            <div style="transform: scale(1.5); transform-origin: center; margin: 40px 0;">
```

<!-- Duplicate the Mermaid diagram for modal (enlarged) -->
```{mermaid}
graph TD
    A[Node A] --> B[Node B]
    B --> C[Node C]
```

```{raw} html
            </div>
        </div>
    </div>
</div>

<script>
function openDiagramModal(modalId, title) {
    const modal = document.getElementById(modalId);
    const modalTitle = document.getElementById(modalId + '-title');
    
    modal.style.display = 'block';
    modalTitle.textContent = title;
    
    document.body.style.overflow = 'hidden';
}

function closeDiagramModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

// Unified Escape key handler
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const modals = document.querySelectorAll('[id*="-modal"]');
        modals.forEach(modal => {
            if (modal.style.display === 'block') {
                modal.style.display = 'none';
                document.body.style.overflow = 'auto';
            }
        });
    }
});
</script>
```

## Key Requirements

1. **Unique IDs**: Each modal must have a unique ID to avoid conflicts
2. **Target by ID**: JavaScript targets specific diagrams by ID for reliability  
3. **Responsive Scaling**: Diagrams scale appropriately for modal viewing
4. **Scroll Support**: Large content scrolls within modal (max-height: 80vh)
5. **Unified Escape Handler**: Single event listener handles all modals

## Styling Features

- **Hover Effects**: Border color changes and subtle background highlight
- **Backdrop Blur**: Modern blur effect behind modal
- **Smooth Transitions**: CSS transitions for professional feel
- **Responsive Width**: Modal adapts to screen size (90-95% width, max 1200-1400px)
- **Close Button**: Styled close button with hover effects

## Browser Support

- Modern browsers with CSS backdrop-filter support
- Fallback to solid background for older browsers
- JavaScript ES6+ features (arrow functions, const/let)

## Usage Examples

See implementations in:
- `docs/about/index.md` - Component architecture diagram
- `docs/about/architecture-overview.md` - Layered architecture diagram
- `docs/guides/omni/gemma3-3n.md` - Training loss and MedPix sample images
- `docs/blogs/gemma3n-blog.md` - Blog training loss image
