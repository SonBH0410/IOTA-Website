var bolb;
// Function to convert base64 to Blob
function base64ToBlob(base64, mimeType) {
  const byteCharacters = atob(base64);
  const byteArrays = [];

  for (let offset = 0; offset < byteCharacters.length; offset += 512) {
    const slice = byteCharacters.slice(offset, offset + 512);
    const byteNumbers = new Array(slice.length);

    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }

    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }

  return new Blob(byteArrays, { type: mimeType });
}

document.getElementById('detect_auto').addEventListener('click', async (event) => {
  event.preventDefault();

  // Check if blob variable exists and is valid
  if (typeof blob === 'undefined' || blob === null) {
    // alert('Please select an image before performing detection!');
    createToast('error', 'fa-solid fa-circle-exclamation', 'Failed', 'Unable to connect to Server');
    return;
  }

  const formData = new FormData();
  formData.append('image', blob);

  try {
    const response = await fetch('http://127.0.0.1:5000/api/detect', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    // Check API response
    const data = await response.json();
    console.log(data); // Check response data

    // Handle JSON if it's an invalid string
    if (typeof data.detections === 'string') {
      // Replace ' with " to make JSON valid
      const jsonString = data.detections.replace(/'/g, '"');
      console.log(jsonString); // Check JSON string

      // Parse JSON string to JavaScript object array
      const detections = JSON.parse(jsonString);
      console.log(detections); // Check JSON object

      const resultInfo = document.getElementById('result_info');
      resultInfo.innerHTML = ''; // Clear all previous content

      // Add <h3> tag with content "Tumor Information"
      const newHeading = document.createElement('h3');
      newHeading.textContent = 'Tumor Information';
      resultInfo.appendChild(newHeading);

      // Decode base64 image and create URL for display
      const base64Image = data.image;
      const imageBlob = base64ToBlob(base64Image, 'image/png');
      const imgUrl = URL.createObjectURL(imageBlob);

      document.getElementById('result').style.display = "block";
      document.getElementById('convertedImage').src = imgUrl;

      if (Array.isArray(detections) && detections.length === 0) {
        document.getElementById('result_none').style.display = "block";
        createToast('error', 'fa-solid fa-circle-exclamation', 'Error', 'No tumor detected');
      } else {
        const fragment = document.createDocumentFragment();
        detections.forEach(detection => {
          const newDiv = document.createElement('div'); 
          newDiv.style.backgroundColor = '#FFFFFF';
          newDiv.style.padding = '10px';
          newDiv.style.margin = '10px 0 0 0';
      
          const coordinateP = document.createElement('p');
          coordinateP.innerHTML = `<b>Detection coordinates:</b> [${detection.box.join(', ')}]`;
          newDiv.appendChild(coordinateP);
      
          const classP = document.createElement('p');
          let classNameText;
          switch (detection.cls) {
            case 0:
              classNameText = "Chocolate cyst";
              break;
            case 1:
              classNameText = "Serous cystadenoma";
              break;
            case 2:
              classNameText = "Teratoma";
              break;
            case 3:
              classNameText = "Theca cell tumor";
              break;
            case 4:
              classNameText = "Simple cysts";
              break;
            case 5:
              classNameText = "Ovary normal";
              break;
            case 6:
              classNameText = "Mucinous cystadenoma";
              break;
            case 7:
              classNameText = "High grade serous cystadenoma";
              break;
            default:
              classNameText = "Unknown";
          }
          classP.innerHTML = `<b>Class:</b> ${classNameText}`;
          newDiv.appendChild(classP);
      
          const confidenceP = document.createElement('p');
          confidenceP.innerHTML = `<b>Prediction confidence:</b> ${detection.conf.toFixed(2)}`;
          newDiv.appendChild(confidenceP);
      
          fragment.appendChild(newDiv);
        });
        const resultInfo = document.getElementById('result_info');
        resultInfo.appendChild(fragment);
        document.getElementById('result_info').style.display = "block";
        createToast('success', 'fa-solid fa-circle-check', 'Success', 'Please see results below');
      }
      
    } else {
      // alert('Invalid JSON data.');
      createToast('error', 'fa-solid fa-circle-exclamation', 'Error', 'Unable to connect to Server');
    }
    const resultDiv = document.getElementById('result');
    // Set TimeOut to ensure the element has been added to DOM before scrolling down
    setTimeout(() => {
      resultDiv.scrollIntoView({ behavior: 'smooth' });
    }, 100);

  } catch (error) {
    console.error('Error:', error);
    // alert('Failed to connect to Server.');
    createToast('error', 'fa-solid fa-circle-exclamation', 'Error', 'Unable to connect to Server');
  }
});

function ekUpload() {
  function Init() {
    console.log("Upload Initialised");

    var fileSelect = document.getElementById('file-upload'),
        fileDrag = document.getElementById('file-drag'),
        submitButton = document.getElementById('submit-button');

    fileSelect.addEventListener('change', fileSelectHandler, false);

    // Is XHR2 available?
    var xhr = new XMLHttpRequest();
    if (xhr.upload) {
      // File Drop
      fileDrag.addEventListener('dragover', fileDragHover, false);
      fileDrag.addEventListener('dragleave', fileDragHover, false);
      fileDrag.addEventListener('drop', fileSelectHandler, false);
    }
  }

  function fileDragHover(e) {
    var fileDrag = document.getElementById('file-drag');

    e.stopPropagation();
    e.preventDefault();

    fileDrag.className = (e.type === 'dragover' ? 'hover' : 'modal-body file-upload');
  }

  function fileSelectHandler(e) {
    // Fetch FileList object
    var files = e.target.files || e.dataTransfer.files;

    // Cancel event and hover styling
    fileDragHover(e);

    // Process all File objects
    for (var i = 0, f; f = files[i]; i++) {
      parseFile(f);
      // Do not upload file here as we want to handle it on button click
      // uploadFile(f);
    }
  }

  // Output
  function output(msg) {
    // Response
    var m = document.getElementById('messages');
    m.innerHTML = msg;
  }

  function parseFile(file) {
    console.log(file.name);
    output('<strong>' + encodeURI(file.name) + '</strong>');

    var fileType = file.type;
    console.log(fileType);
    var imageName = file.name;

    var isGood = (/\.(?=gif|jpg|png|jpeg)/gi).test(imageName);
    if (isGood) {
      console.log("Valid file: " + file.name);
      document.getElementById('start').classList.add("hidden");
      document.getElementById('response').classList.remove("hidden");
      document.getElementById('notimage').classList.add("hidden");

      // Thumbnail Preview
      document.getElementById('file-image').classList.remove("hidden");
      document.getElementById('file-image').src = URL.createObjectURL(file);
      document.getElementById('button-del-search').classList.remove("hidden");
      document.querySelector('.uploader label').style.padding = '2rem 1.5rem';

      var fileReader = new FileReader();
      fileReader.onloadend = function() {
        blob = new Blob([fileReader.result]);
      };
      fileReader.readAsArrayBuffer(file);
    } else {
      document.getElementById('file-image').classList.add("hidden");
      document.getElementById('notimage').classList.remove("hidden");
      document.getElementById('start').classList.remove("hidden");
      document.getElementById('response').classList.add("hidden");
      document.getElementById("file-upload-form").reset();
    }
  }

  function setProgressMaxValue(e) {
    var pBar = document.getElementById('file-progress');

    if (e.lengthComputable) {
      pBar.max = e.total;
    }
  }

  function updateFileProgress(e) {
    var pBar = document.getElementById('file-progress');

    if (e.lengthComputable) {
      pBar.value = e.loaded;
    }
  }

  function uploadFile(file) {
    // This function is now redundant as the file upload is handled on button click
  }

  // Check for the various File API support.
  if (window.File && window.FileList && window.FileReader) {
    Init();
  } else {
    // document.getElementById('file-drag').style.display = 'none';
  }
}

ekUpload();
