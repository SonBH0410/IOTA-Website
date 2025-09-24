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

    // success.onclick = function(){
    //     let type = 'success';
    //     let icon = 'fa-solid fa-circle-check';
    //     let title = 'Success';
    //     let text = 'This is a success toast.';
    //     createToast(type, icon, title, text);
    // }
    // error.onclick = function(){
    //     let type = 'error';
    //     let icon = 'fa-solid fa-circle-exclamation';
    //     let title = 'Error';
    //     let text = 'Please complete all information';
    //     createToast(type, icon, title, text);
    // }
    // warning.onclick = function(){
    //     let type = 'warning';
    //     let icon = 'fa-solid fa-triangle-exclamation';
    //     let title = 'Warning';
    //     let text = 'This is a warning toast.';
    //     createToast(type, icon, title, text);
    // }
    // info.onclick = function(){
    //     let type = 'info';
    //     let icon = 'fa-solid fa-circle-info';
    //     let title = 'Info';
    //     let text = 'This is a info toast.';
    //     createToast(type, icon, title, text);
    // }
  
function resetCheckboxes() {
  window.location.href = window.location.href;
}
  // Left ovary
  var leftovarydisplay = document.getElementsByClassName("left-ovary");
  var leftOvaryCheck = false;
  var leftovary = document.querySelector('input[name="checkbox-left-ovary"]');
  leftovary.addEventListener("change",function(){
    leftOvaryCheck = leftovary.checked;
    if(leftOvaryCheck == true){
      for (let i = 0; i < leftovarydisplay.length; i++) {
        leftovarydisplay[i].style.display = "block";
      }
    }else{
      for (let i = 0; i < leftovarydisplay.length; i++) {
        leftovarydisplay[i].style.display = "none";
      }
    }
  });

  //Right ovary
  var rightOvaryCheck = false;
  var rightOvary = document.querySelector('input[name="checkbox-right-ovary"]');
  var rightOvaryDisplay = document.getElementsByClassName("right-ovary");

  rightOvary.addEventListener("change",function(){
    rightOvaryCheck = rightOvary.checked;
    if(rightOvaryCheck == true){
      for(let i = 0;i < rightOvaryDisplay.length; i++){
        rightOvaryDisplay[i].style.display = "block";
      }
    }else{
      for(let i = 0;i<rightOvaryDisplay.length; i++){
        rightOvaryDisplay[i].style.display = "none";
      }
    }
  })


var fullname = document.querySelector('input[name="patient_fullname"]').value;
var patientID = document.querySelector('input[name="patient_id"]').value;

var irregularSolidTumor_yes_left = document.querySelector('input[name="irregular_solid_tumor_yes_left"]');
var irregularSolidTumor_no_left = document.querySelector('input[name="irregular_solid_tumor_no_left"]');

var unilocular_yes_left = document.querySelector('input[name="unilocular_yes_left"]');
var unilocular_no_left = document.querySelector('input[name="unilocular_no_left"]');

var ascites_yes_left = document.querySelector('input[name="presence_of_ascites_yes_left"]');
var ascites_no_left = document.querySelector('input[name="presence_of_ascites_no_left"]');

var solidComponents_yes_left = document.querySelector('input[name="solid_components_yes_left"]');
var solidComponents_no_left = document.querySelector('input[name="solid_components_no_left"]');

var papillaryStructures_yes_left = document.querySelector('input[name="papillary_structures_yes_left"]');
var papillaryStructures_no_left = document.querySelector('input[name="papillary_structures_no_left"]');

var acousticShadows_yes_left = document.querySelector('input[name="acoustic_shadows_presence_yes_left"]');
var acousticShadows_no_left = document.querySelector('input[name="acoustic_shadows_presence_no_left"]');

var irregularMultilocularTumor_yes_left = document.querySelector('input[name="irregular_soild_tumor_yes_left"]');
var irregularMultilocularTumor_no_left = document.querySelector('input[name="irregular_soild_tumor_no_left"]');

var smothMultilocularTumor_yes_left = document.querySelector('input[name="multilocular_tumor_yes_left"]');
var smothMultilocularTumor_no_left = document.querySelector('input[name="multilocular_tumor_no_left"]');

var strongBloodFlow_yes_left = document.querySelector('input[name="strong_blood_flow_yes_left"]');
var strongBloodFlow_no_left = document.querySelector('input[name="strong_blood_flow_no_left"]');

var noBloodFlow_yes_left = document.querySelector('input[name="no_blood_flow_yes_left"]');
var noBloodFlow_no_left = document.querySelector('input[name="no_blood_flow_no_left"]');

var irregularSolidTumor_yes_right = document.querySelector('input[name="irregular_solid_tumor_yes_right"]');
var irregularSolidTumor_no_right = document.querySelector('input[name="irregular_solid_tumor_no_right"]');

var unilocular_yes_right = document.querySelector('input[name="unilocular_yes_right"]');
var unilocular_no_right = document.querySelector('input[name="unilocular_no_right"]');

var ascites_yes_right = document.querySelector('input[name="presence_of_ascites_yes_right"]');
var ascites_no_right = document.querySelector('input[name="presence_of_ascites_no_right"]');

var solidComponents_yes_right = document.querySelector('input[name="solid_components_yes_right"]');
var solidComponents_no_right = document.querySelector('input[name="solid_components_no_right"]');

var papillaryStructures_yes_right = document.querySelector('input[name="papillary_structures_yes_right"]');
var papillaryStructures_no_right = document.querySelector('input[name="papillary_structures_no_right"]');

var acousticShadows_yes_right = document.querySelector('input[name="acoustic_shadows_presence_yes_right"]');
var acousticShadows_no_right = document.querySelector('input[name="acoustic_shadows_presence_no_right"]');

var irregularMultilocularTumor_yes_right = document.querySelector('input[name="irregular_soild_tumor_yes_right"]');
var irregularMultilocularTumor_no_right = document.querySelector('input[name="irregular_soild_tumor_no_right"]');

var smothMultilocularTumor_yes_right = document.querySelector('input[name="multilocular_tumor_yes_right"]');
var smothMultilocularTumor_no_right = document.querySelector('input[name="multilocular_tumor_no_right"]');

var strongBloodFlow_yes_right = document.querySelector('input[name="strong_blood_flow_yes_right"]');
var strongBloodFlow_no_right = document.querySelector('input[name="strong_blood_flow_no_right"]');
 
var noBloodFlow_yes_right = document.querySelector('input[name="no_blood_flow_yes_right"]');
var noBloodFlow_no_right = document.querySelector('input[name="no_blood_flow_no_right"]');

// ---------------------checkbox eventlistener-------------------
// Button No B-rules Left Ovary
  // 1. Irregular solid tumor 
  var irregularSolidTumorCheck_no_left = false;
  irregularSolidTumor_no_left.addEventListener("change", function(){
    irregularSolidTumorCheck_no_left = irregularSolidTumor_no_left.checked;
  });

  // 2. Unilocular
  var unilocularCheck_no_left = false;
  unilocular_no_left.addEventListener("change",function(){
    unilocularCheck_no_left = unilocular_no_left.checked;
  })

  // 3. Presence of ascites
  var ascitesCheck_no_left = false;
  ascites_no_left.addEventListener("change",function(){
    ascitesCheck_no_left = ascites_no_left.checked;
  })

  // 4. Presence of solid componenets
  var solidComponentsCheck_no_left = false;
  solidComponents_no_left.addEventListener("change",function(){
    solidComponentsCheck_no_left = solidComponents_no_left.checked;
  })

  // 5. At least four papillary structures
  var papillaryStructuresCheck_no_left = false;
  papillaryStructures_no_left.addEventListener("change",function(){
    papillaryStructuresCheck_no_left = papillaryStructures_no_left.checked;
  })

  // 6. Precence of acoustic
  var acousticShadowsCheck_no_left = false;
  acousticShadows_no_left.addEventListener("change",function(){
    acousticShadowsCheck_no_left = acousticShadows_no_left.checked;
  }) 

  // 7. Irregular multilocular soild tumor with largest diameter >= 100
  var irregularMultilocularTumorCheck_no_left = false;
  irregularMultilocularTumor_no_left.addEventListener("change",function(){
    irregularMultilocularTumorCheck_no_left = irregularMultilocularTumor_no_left.checked;
  }) 

  // 8. Smooth multilocular tumor with largest diameter < 100mm
  var smothMultilocularTumorCheck_no_left = false;
  smothMultilocularTumor_no_left.addEventListener("change",function(){
    smothMultilocularTumorCheck_no_left = smothMultilocularTumor_no_left.checked;
  }) 

  // 9. Very strong blood flow (color score 4):
  var strongBloodFlowCheck_no_left= false;
  strongBloodFlow_no_left.addEventListener("change",function(){
    strongBloodFlowCheck_no_left = strongBloodFlow_no_left.checked;
  }) 

  // 10. No blood flow (color score 1)
  var noBloodFlowCheck_no_left = false;
  noBloodFlow_no_left.addEventListener("change",function(){
    noBloodFlowCheck_no_left = noBloodFlow_no_left.checked;
  }) 

  //Button Yes B-rules Left Ovary
  // 1. Irregular solid tumor 
  var irregularSolidTumorCheck_yes_left = false;
  irregularSolidTumor_yes_left.addEventListener("change", function(){
    irregularSolidTumorCheck_yes_left = irregularSolidTumor_yes_left.checked;
  });

  // 2. Unilocular
  var unilocularCheck_yes_left = false;
  unilocular_yes_left.addEventListener("change",function(){
    unilocularCheck_yes_left = unilocular_yes_left.checked;
  });

  // 3. Presence of ascites
  var ascitesCheck_yes_left = false;
  ascites_yes_left.addEventListener("change",function(){
    ascitesCheck_yes_left = ascites_yes_left.checked;
  });

  // 4. Presence of solid componenets
  var solidComponentsCheck_yes_left = false;
  solidComponents_yes_left.addEventListener("change",function(){
    solidComponentsCheck_yes_left = solidComponents_yes_left.checked;
  });

  // 5. At least four papillary structures
  var papillaryStructuresCheck_yes_left = false;
  papillaryStructures_yes_left.addEventListener("change",function(){
    papillaryStructuresCheck_yes_left = papillaryStructures_yes_left.checked;
  });

  // 6. Precence of acoustic
  var acousticShadowsCheck_yes_left = false;
  acousticShadows_yes_left.addEventListener("change",function(){
    acousticShadowsCheck_yes_left = acousticShadows_yes_left.checked;
  }); 

  // 7. Irregular multilocular soild tumor with largest diameter >= 100
  var irregularMultilocularTumorCheck_yes_left = false;
  irregularMultilocularTumor_yes_left.addEventListener("change",function(){
    irregularMultilocularTumorCheck_yes_left = irregularMultilocularTumor_yes_left.checked;
  });

  // 8. Smooth multilocular tumor with largest diameter < 100mm
  var smothMultilocularTumorCheck_yes_left = false;
  smothMultilocularTumor_yes_left.addEventListener("change",function(){
    smothMultilocularTumorCheck_yes_left = smothMultilocularTumor_yes_left.checked;
  });

  // 9. Very strong blood flow (color score 4):
  var strongBloodFlowCheck_yes_left = false;
  strongBloodFlow_yes_left.addEventListener("change",function(){
    strongBloodFlowCheck_yes_left = strongBloodFlow_yes_left.checked;
  });

  // 10. No blood flow (color score 1)
  var noBloodFlowCheck_yes_left = false;
  noBloodFlow_yes_left.addEventListener("change",function(){
    noBloodFlowCheck_yes_left = noBloodFlow_yes_left.checked;
  });
  // ----------------------------------------------------------
  // Button No B-rules Right Ovary
  // 1. Irregular solid tumor 
  var irregularSolidTumorCheck_no_right = false;
  irregularSolidTumor_no_right.addEventListener("change", function(){
    irregularSolidTumorCheck_no_right = irregularSolidTumor_no_right.checked;
  });

  // 2. Unilocular
  var unilocularCheck_no_right = false;
  unilocular_no_right.addEventListener("change",function(){
    unilocularCheck_no_right = unilocular_no_right.checked;
  })

  // 3. Presence of ascites
  var ascitesCheck_no_right = false;
  ascites_no_right.addEventListener("change",function(){
    ascitesCheck_no_right = ascites_no_right.checked;
  })

  // 4. Presence of solid componenets
  var solidComponentsCheck_no_right = false;
  solidComponents_no_right.addEventListener("change",function(){
    solidComponentsCheck_no_right = solidComponents_no_right.checked;
  })

  // 5. At least four papillary structures
  var papillaryStructuresCheck_no_right = false;
  papillaryStructures_no_right.addEventListener("change",function(){
    papillaryStructuresCheck_no_right = papillaryStructures_no_right.checked;
  })

  // 6. Precence of acoustic
  var acousticShadowsCheck_no_right = false;
  acousticShadows_no_right.addEventListener("change",function(){
    acousticShadowsCheck_no_right = acousticShadows_no_right.checked;
  }) 

  // 7. Irregular multilocular soild tumor with largest diameter >= 100
  var irregularMultilocularTumorCheck_no_right = false;
  irregularMultilocularTumor_no_right.addEventListener("change",function(){
    irregularMultilocularTumorCheck_no_right = irregularMultilocularTumor_no_right.checked;
  }) 

  // 8. Smooth multilocular tumor with largest diameter < 100mm
  var smothMultilocularTumorCheck_no_right = false;
  smothMultilocularTumor_no_right.addEventListener("change",function(){
    smothMultilocularTumorCheck_no_right = smothMultilocularTumor_no_right.checked;
  }) 

  // 9. Very strong blood flow (color score 4):
  var strongBloodFlowCheck_no_right= false;
  strongBloodFlow_no_right.addEventListener("change",function(){
    strongBloodFlowCheck_no_right = strongBloodFlow_no_right.checked;
  }) 

  // 10. No blood flow (color score 1)
  var noBloodFlowCheck_no_right = false;
  noBloodFlow_no_right.addEventListener("change",function(){
    noBloodFlowCheck_no_right = noBloodFlow_no_right.checked;
  }) 

  //Button Yes B-rules right Ovary
  // 1. Irregular solid tumor 
  var irregularSolidTumorCheck_yes_right = false;
  irregularSolidTumor_yes_right.addEventListener("change", function(){
    irregularSolidTumorCheck_yes_right = irregularSolidTumor_yes_right.checked;
  });

  // 2. Unilocular
  var unilocularCheck_yes_right = false;
  unilocular_yes_right.addEventListener("change",function(){
    unilocularCheck_yes_right = unilocular_yes_right.checked;
  });

  // 3. Presence of ascites
  var ascitesCheck_yes_right = false;
  ascites_yes_right.addEventListener("change",function(){
    ascitesCheck_yes_right = ascites_yes_right.checked;
  });

  // 4. Presence of solid componenets
  var solidComponentsCheck_yes_right = false;
  solidComponents_yes_right.addEventListener("change",function(){
    solidComponentsCheck_yes_right = solidComponents_yes_right.checked;
  });

  // 5. At least four papillary structures
  var papillaryStructuresCheck_yes_right = false;
  papillaryStructures_yes_right.addEventListener("change",function(){
    papillaryStructuresCheck_yes_right = papillaryStructures_yes_right.checked;
  });

  // 6. Precence of acoustic
  var acousticShadowsCheck_yes_right = false;
  acousticShadows_yes_right.addEventListener("change",function(){
    acousticShadowsCheck_yes_right = acousticShadows_yes_right.checked;
  }); 

  // 7. Irregular multilocular soild tumor with largest diameter >= 100
  var irregularMultilocularTumorCheck_yes_right = false;
  irregularMultilocularTumor_yes_right.addEventListener("change",function(){
    irregularMultilocularTumorCheck_yes_right = irregularMultilocularTumor_yes_right.checked;
  });

  // 8. Smooth multilocular tumor with largest diameter < 100mm
  var smothMultilocularTumorCheck_yes_right = false;
  smothMultilocularTumor_yes_right.addEventListener("change",function(){
    smothMultilocularTumorCheck_yes_right = smothMultilocularTumor_yes_right.checked;
  });

  // 9. Very strong blood flow (color score 4):
  var strongBloodFlowCheck_yes_right = false;
  strongBloodFlow_yes_right.addEventListener("change",function(){
    strongBloodFlowCheck_yes_right = strongBloodFlow_yes_right.checked;
  });

  // 10. No blood flow (color score 1)
  var noBloodFlowCheck_yes_right = false;
  noBloodFlow_yes_right.addEventListener("change",function(){
    noBloodFlowCheck_yes_right = noBloodFlow_yes_right.checked;
  });
var coutCheck1 = 0;
var coutCheck2 = 0;
// ------------------------SHOW RESULTS----------------------------
function checkBoxResult_Left(){
  coutCheck1 = 0
  coutCheck2 = 0;
  if(!irregularSolidTumorCheck_yes_left &&
    !ascitesCheck_yes_left &&
    !papillaryStructuresCheck_yes_left &&
    !irregularMultilocularTumorCheck_yes_left &&
    !strongBloodFlowCheck_yes_left){
      ++coutCheck1;
    }
  if(
    unilocularCheck_yes_left ||
    solidComponentsCheck_yes_left ||
    acousticShadowsCheck_yes_left ||
    smothMultilocularTumorCheck_yes_left ||
    noBloodFlowCheck_yes_left){
      ++coutCheck1;
    }
    if(coutCheck1 == 2){
      return 1;
    }

  if(irregularSolidTumorCheck_yes_left ||
    ascitesCheck_yes_left ||
    papillaryStructuresCheck_yes_left ||
    irregularMultilocularTumorCheck_yes_left ||
    strongBloodFlowCheck_yes_left){
      ++coutCheck2;
    }
  if(
    !unilocularCheck_yes_left &&
    !solidComponentsCheck_yes_left &&
    !acousticShadowsCheck_yes_left &&
    !smothMultilocularTumorCheck_yes_left &&
    !noBloodFlowCheck_yes_left){
      ++coutCheck2;
    }
  if(coutCheck2 == 2){
    return 2;
  }
  return 0;

}
function checkBoxResult_Right(){
  coutCheck1 = 0
  coutCheck2 = 0;
  if(!irregularSolidTumorCheck_yes_right &&
    !ascitesCheck_yes_right &&
    !papillaryStructuresCheck_yes_right &&
    !irregularMultilocularTumorCheck_yes_right &&
    !strongBloodFlowCheck_yes_right){
      ++coutCheck1;
    }
  if(
    unilocularCheck_yes_right ||
    solidComponentsCheck_yes_right ||
    acousticShadowsCheck_yes_right ||
    smothMultilocularTumorCheck_yes_right ||
    noBloodFlowCheck_yes_right){
      ++coutCheck1;
    }
    if(coutCheck1 == 2){
      return 1;
    }

  if(irregularSolidTumorCheck_yes_right ||
    ascitesCheck_yes_right ||
    papillaryStructuresCheck_yes_right ||
    irregularMultilocularTumorCheck_yes_right ||
    strongBloodFlowCheck_yes_right){
      ++coutCheck2;
    }
  if(
    !unilocularCheck_yes_right &&
    !solidComponentsCheck_yes_right &&
    !acousticShadowsCheck_yes_right &&
    !smothMultilocularTumorCheck_yes_right &&
    !noBloodFlowCheck_yes_right){
      ++coutCheck2;
    }
  if(coutCheck2 == 2){
    return 2;
  }
  return 0;
}
function checkResults() {
   // alert(irregularSolidTumorCheck + " " + unilocularCheck + " " + ascitesCheck +" " + solidComponentsCheck +" "+papillaryStructuresCheck +" "+ acousticShadowsCheck + " " + irregularMultilocularTumorCheck + " " + smothMultilocularTumorCheck + " " + strongBloodFlowCheck + " " + noBloodFlowCheck);
    // --------------------Diagnosis------------------
    var checkboxL  = document.querySelector('input[name="checkbox-left-ovary"]');
    if(checkboxL.checked){
      document.getElementById("group-text-diagnosis").style.display ="block";
      document.getElementById("text-diagnosis-results-left").style.display ="block";
      document.getElementById("text-diagnosis-comments-left").style.display ="block";
      var checkPoint_Left = checkBoxResult_Left();
      if(checkPoint_Left== 1){
          addAlert("Left ovary: According to Ultrasound Findings Information, THERE IS A LOW RISK OF MALIGNANCY, pursuant to the Simple Rules from the International Ovarian Tumor Analysis (IOTA) Group.","text-diagnosis-results-left");
          addAlert("Left ovary: Only features typical for benign tumors (B-features) apply, with none feature typical for malignant tumors (M-features).","text-diagnosis-comments-left");
        }
      if(checkPoint_Left === 2){
            addAlert("Left ovary: According to Ultrasound Findings Information, THERE IS A HIGH RISK OF MALIGNANCY, pursuant to the Simple Rules from the International Ovarian Tumor Analysis (IOTA) Group.","text-diagnosis-results-left");
            addAlert("Left ovary: Only features typical for malignant tumors (M-features) apply, with none feature typical for benign tumors (B-features).","text-diagnosis-comments-left");
        }
      if(checkPoint_Left === 0){
            addAlert("Left ovary: According to Ultrasound Findings Information, THERE IS AN UNDETERMINED RISK OF MALIGNANCY, pursuant to the Simple Rules from the International Ovarian Tumor Analysis (IOTA) Group.","text-diagnosis-results-left");
            addAlert("Left ovary: No features apply, or both B-features and M-features apply.","text-diagnosis-comments-left");
        }
    } 
    var checkboxR  = document.querySelector('input[name="checkbox-right-ovary"]');
    if(checkboxR.checked){
      document.getElementById("group-text-diagnosis").style.display ="block";
      document.getElementById("text-diagnosis-results-right").style.display ="block";
      document.getElementById("text-diagnosis-comments-right").style.display ="block";
    var checkPoint_Right = checkBoxResult_Right();
    if(checkPoint_Right== 1){
        addAlert("Right ovary: According to Ultrasound Findings Information, THERE IS A LOW RISK OF MALIGNANCY, pursuant to the Simple Rules from the International Ovarian Tumor Analysis (IOTA) Group.","text-diagnosis-results-right");
        addAlert("Right ovary: Only features typical for benign tumors (B-features) apply, with none feature typical for malignant tumors (M-features).","text-diagnosis-comments-right");
      }
    if(checkPoint_Right === 2){
          addAlert("Right ovary: According to Ultrasound Findings Information, THERE IS A HIGH RISK OF MALIGNANCY, pursuant to the Simple Rules from the International Ovarian Tumor Analysis (IOTA) Group.","text-diagnosis-results-right");
          addAlert("Right ovary: Only features typical for malignant tumors (M-features) apply, with none feature typical for benign tumors (B-features).","text-diagnosis-comments-right");
      }
    if(checkPoint_Right === 0){
          addAlert("Right ovary: According to Ultrasound Findings Information, THERE IS AN UNDETERMINED RISK OF MALIGNANCY, pursuant to the Simple Rules from the International Ovarian Tumor Analysis (IOTA) Group.","text-diagnosis-results-right");
          addAlert("Right ovary: No features apply, or both B-features and M-features apply.","text-diagnosis-comments-right");
      }
    }
    if(checkboxL.checked && checkboxR.checked){
      createToast('success', 'fa-solid fa-circle-check', 'Success', 'Please see the results below');
      document.getElementById('group-text-diagnosis').scrollIntoView({ behavior: 'smooth' });
    }else if(checkboxL.checked){
      createToast('success', 'fa-solid fa-circle-check', 'Success', 'Please see the results below');
      document.getElementById('group-text-diagnosis').scrollIntoView({ behavior: 'smooth' });
    }else if(checkboxR.checked){
      createToast('success', 'fa-solid fa-circle-check', 'Success', 'Please see the results below');
      document.getElementById('group-text-diagnosis').scrollIntoView({ behavior: 'smooth' });
    }

    if(!checkboxL.checked && !checkboxR.checked){
      createToast('warning', 'fa-solid fa-triangle-exclamation', 'Warning', 'No information was entered');
    }
}


// --------------------Disable Checkbox -------------------
function setupCheckboxDependency(checkbox1, checkbox2) {
  checkbox1.addEventListener("change", function() {
    checkbox2.disabled = checkbox1.checked;
  });
  checkbox2.addEventListener("change", function() {
    checkbox1.disabled = checkbox2.checked;
  });
}
setupCheckboxDependency(unilocular_yes_left, unilocular_no_left);
setupCheckboxDependency(solidComponents_yes_left, solidComponents_no_left);
setupCheckboxDependency(acousticShadows_yes_left, acousticShadows_no_left);
setupCheckboxDependency(smothMultilocularTumor_yes_left, smothMultilocularTumor_no_left);
setupCheckboxDependency(noBloodFlow_yes_left, noBloodFlow_no_left);

setupCheckboxDependency(irregularSolidTumor_yes_left, irregularSolidTumor_no_left);
setupCheckboxDependency(irregularMultilocularTumor_yes_left, irregularMultilocularTumor_no_left);
setupCheckboxDependency(papillaryStructures_yes_left, papillaryStructures_no_left);
setupCheckboxDependency(strongBloodFlow_yes_left, strongBloodFlow_no_left);
setupCheckboxDependency(ascites_yes_left, ascites_no_left);

setupCheckboxDependency(unilocular_yes_right, unilocular_no_right);
setupCheckboxDependency(solidComponents_yes_right, solidComponents_no_right);
setupCheckboxDependency(acousticShadows_yes_right, acousticShadows_no_right);
setupCheckboxDependency(smothMultilocularTumor_yes_right, smothMultilocularTumor_no_right);
setupCheckboxDependency(noBloodFlow_yes_right, noBloodFlow_no_right);

setupCheckboxDependency(irregularSolidTumor_yes_right, irregularSolidTumor_no_right);
setupCheckboxDependency(irregularMultilocularTumor_yes_right, irregularMultilocularTumor_no_right);
setupCheckboxDependency(papillaryStructures_yes_right, papillaryStructures_no_right);
setupCheckboxDependency(strongBloodFlow_yes_right, strongBloodFlow_no_right);
setupCheckboxDependency(ascites_yes_right, ascites_no_right);


// -------------------Add textContent--------------
function addAlert(textContent, id) {
  var paragraph = document.getElementById(id);
  paragraph.textContent = textContent;
}
function deleteAlert(id) {
  var paragraph = document.getElementById(id);
  paragraph.textContent = "";
}

