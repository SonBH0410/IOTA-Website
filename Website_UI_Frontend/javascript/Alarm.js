let notifications = document.querySelector('.notifications');
let success = document.getElementById('success');
let error = document.getElementById('error');
let warning = document.getElementById('warning');
let info = document.getElementById('info');

function createToast(type, icon, title, text) {
    // Clear previous notifications
    notifications.innerHTML = ''; // Clear all previous notifications

    let newToast = document.createElement('div');
    newToast.innerHTML = `
        <div class="toast ${type}">
            <i class="${icon}"></i>
            <div class="content">
                <div class="title">${title}</div>
                <span>${text}</span>
            </div>
            <i class="fa-solid fa-xmark" onclick="removeToast(this.parentElement)"></i>
        </div>`;
    notifications.appendChild(newToast);

    // Set timeout to remove the toast after 5 seconds
    newToast.timeOut = setTimeout(() => newToast.remove(), 5000);

    // Add an event listener to reset the timeout when clicked
    newToast.addEventListener('click', () => {
        clearTimeout(newToast.timeOut);
        newToast.timeOut = setTimeout(() => newToast.remove(), 5000);
    });
}

function removeToast(toastElement) {
    clearTimeout(toastElement.timeOut);  // Clear the timeout when manually closing the toast
    toastElement.remove();
}
