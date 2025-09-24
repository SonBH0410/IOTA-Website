// Call the function when needed
var cout = 0;
var checkLeftOvary = true;
var checkRightOvary = true;
var resultsDiagnosis = document.getElementById("results-diagnosis");
resultsDiagnosis.style.display = "none";

var resultLeft = document.getElementsByClassName('text-alert-left');
var resultRight = document.getElementsByClassName('text-alert-right');

var checkboxL  = document.querySelector('input[name="checkbox-left-ovary"]');
var checkboxR  = document.querySelector('input[name="checkbox-right-ovary"]');

for (var i = 0; i < resultLeft.length; i++) {
  resultLeft[i].style.display = "none";
}

for (var i = 0; i < resultRight.length; i++) {
  resultRight[i].style.display = "none";
}

function resetCheckboxes_LR() {
  window.location.href = window.location.href;
}

function checkResults_LR() {
  cout = 0;
   //=======================================================================================
  //Left Ovary
  //=======================================================================================
  if(checkboxL.checked){
  var agePatientValue = parseInt(document.querySelector('input[name="age_patient_left"]').value);
  var ascitesSelectValue = parseInt(document.getElementById("ascitesSelectLeft").value);
  var bloodFlowSelectValue = parseInt(document.getElementById("bloodFlowSelectLeft").value);
  var solidComponentsValue = parseInt(document.querySelector('input[name="solid_componets_left"]').value);
  var cystWallsValue = parseInt(document.getElementById('cystWallLeft').value);
  var acousticShadowsValue = parseInt(document.getElementById('acousticShadowsLeft').value);
  var historyOvarianValue = parseInt(document.getElementById('historyOvarianLeft').value);
  var hormonalTherapyValue = parseInt(document.getElementById('hormonalTherapyLeft').value);
  var lesionDiameterValue = parseInt(document.querySelector('input[name="lesionDiameterLeft"]').value);
  var painDuringExaminationValue = parseInt(document.getElementById('painDuringExaminationLeft').value);
  var purelySolidTumorValue = parseInt(document.getElementById('purelySolidTumorLeft').value);
  var colorScoreValue = parseInt(document.getElementById('colorScoreLeft').value);

  if(!agePatientValue){
    var ageCheck = document.getElementById("ageCheckLeft");
    ageCheck.textContent = "";
    var iconElement = document.createElement("i");
    iconElement.className = "bi bi-exclamation-diamond-fill";
    ageCheck.appendChild(iconElement);
    var newParagraph = document.createElement("p");
    newParagraph.textContent = " Not null";
    newParagraph.style.color ="#D9534F";
    ageCheck.appendChild(newParagraph);
    checkLeftOvary = false;
    if(cout == 0){
      document.querySelector('input[name="age_patient_left"]').focus();
      ++cout;
      createToast('error', 'fa-solid fa-circle-exclamation', 'Error', 'Please complete all information');
    }
  }else{
    var ageCheck = document.getElementById("ageCheckLeft");
    ageCheck.textContent = "";
    checkLeftOvary = true;
  }

  if(!solidComponentsValue){
    var solidCheck = document.getElementById("solidCheckLeft");
    solidCheck.textContent = "";
    var iconElement = document.createElement("i");
    iconElement.className = "bi bi-exclamation-diamond-fill";
    solidCheck.appendChild(iconElement);
    var newParagraph = document.createElement("p");
    newParagraph.textContent = " Not null";
    newParagraph.style.color ="#D9534F";
    solidCheck.appendChild(newParagraph);
    checkLeftOvary = false;
    if(cout == 0){
      document.querySelector('input[name="solid_componets_left"]').focus();
      ++cout;
      createToast('error', 'fa-solid fa-circle-exclamation', 'Error', 'Please complete all information');
    }
  }else{
    var solidCheck = document.getElementById("solidCheckLeft");
    solidCheck.textContent = "";
    checkLeftOvary = true;
  }

  if(!lesionDiameterValue){
    var lesionCheck = document.getElementById("lesionCheckLeft");
    lesionCheck.textContent = "";
    var iconElement = document.createElement("i");
    iconElement.className = "bi bi-exclamation-diamond-fill";
    lesionCheck.appendChild(iconElement);
    var newParagraph = document.createElement("p");
    newParagraph.textContent = " Not null";
    newParagraph.style.color ="#D9534F";
    lesionCheck.appendChild(newParagraph);
    checkLeftOvary = false;
    if(cout == 0){
      document.querySelector('input[name="lesionDiameterLeft"]').focus();
      ++cout;
      createToast('error', 'fa-solid fa-circle-exclamation', 'Error', 'Please complete all information');
    }
  }else{
    var lesionCheck = document.getElementById("lesionCheckLeft");
    lesionCheck.textContent = "";
    checkLeftOvary = true;
  }
  
  if(checkLeftOvary == true){
      resultsDiagnosis.style.display = "block";

    for (var i = 0; i < resultLeft.length; i++) {
      resultLeft[i].style.display = "block";
    }

    var z1_left = -6.7468 + 0.0326 * agePatientValue + 1.5513 * ascitesSelectValue + 1.1737 * bloodFlowSelectValue + 0.0496 * solidComponentsValue + 1.1421 * cystWallsValue - 2.3550 * acousticShadowsValue + 1.5985 * historyOvarianValue - 0.9983 * hormonalTherapyValue + 0.00841 * lesionDiameterValue - 0.8577 * painDuringExaminationValue + 0.9281 * purelySolidTumorValue + 0.4916 * colorScoreValue;
    var LR1_left = 1 / (1 + Math.exp(-z1_left));
    var z2_left = -5.3718 + 0.0354*agePatientValue + 1.6159*ascitesSelectValue + 1.1768*bloodFlowSelectValue + 0.0697*solidComponentsValue + 0.9586*cystWallsValue - 2.9486*acousticShadowsValue;
    var LR2_left = 1 / (1 + Math.exp(-z2_left));
    
    var LR1LeftOvary = document.getElementById("LR1_Result_LeftOvary");
    LR1LeftOvary.innerHTML = (LR1_left*100).toFixed(2);

    var LR2LeftOvary = document.getElementById("LR2_Result_LeftOvary");
    LR2LeftOvary.innerHTML = (LR2_left*100).toFixed(2);

    google.charts.load('current', {'packages':['corechart']});
    google.charts.setOnLoadCallback(drawChart_LR1Left);

        function drawChart_LR1Left() {

          var data = google.visualization.arrayToDataTable([
            ['Task', 'Hours per Day'],
            ['Benign (%)', 1 - LR1_left],
            ['Manligant (%)', LR1_left]
          ]);

          var options = {
            title: 'LR1'
          };

          var chart = new google.visualization.PieChart(document.getElementById('piechart_LR1Left'));
          chart.draw(data, options);
        }
    
    google.charts.load('current', {'packages':['corechart']});
    google.charts.setOnLoadCallback(drawChart_LR2Left);

        function drawChart_LR2Left() {

          var data = google.visualization.arrayToDataTable([
            ['Task', 'Hours per Day'],
            ['Benign (%)', 1 - LR2_left],
            ['Manligant (%)', LR2_left]
          ]);

          var options = {
            title: 'LR2'
          };

          var chart = new google.visualization.PieChart(document.getElementById('piechart_LR2Left'));
          chart.draw(data, options);
        }
    }
  }
  //=======================================================================================
  //Right Ovary
  //=======================================================================================
  if(checkboxR.checked){
    var agePatientValueRight = parseInt(document.querySelector('input[name="age_patient"]').value);
    var ascitesSelectValueRight = parseInt(document.getElementById("ascitesSelect").value);
    var bloodFlowSelectValueRight = parseInt(document.getElementById("bloodFlowSelect").value);
    var solidComponentsValueRight = parseInt(document.querySelector('input[name="solid_componets_right"]').value);
    var cystWallsValueRight = parseInt(document.getElementById('cystWall').value);
    var acousticShadowsValueRight = parseInt(document.getElementById('acousticShadows').value);
    var historyOvarianValueRight = parseInt(document.getElementById('historyOvarian').value);
    var hormonalTherapyValueRight = parseInt(document.getElementById('hormonalTherapy').value);
    var lesionDiameterValueRight= parseInt(document.querySelector('input[name="lesionDiameter"]').value);
    var painDuringExaminationValueRight = parseInt(document.getElementById('painDuringExamination').value);
    var purelySolidTumorValueRight = parseInt(document.getElementById('purelySolidTumor').value);
    var colorScoreValueRight = parseInt(document.getElementById('colorScore').value);
    
    if(!agePatientValueRight){
      var ageCheck = document.getElementById("ageCheckRight");
      ageCheck.textContent = "";
      var iconElement = document.createElement("i");
      iconElement.className = "bi bi-exclamation-diamond-fill";
      ageCheck.appendChild(iconElement);
      var newParagraph = document.createElement("p");
      newParagraph.textContent = " Not null";
      newParagraph.style.color ="#D9534F";
      ageCheck.appendChild(newParagraph);
      checkRightOvary = false;
      if(cout == 0){
        document.querySelector('input[name="age_patient"]').focus();
        ++cout;
        createToast('error', 'fa-solid fa-circle-exclamation', 'Error', 'Please complete all information');
      }
    }else{
      var ageCheck = document.getElementById("ageCheckRight");
      ageCheck.textContent = "";
      checkRightOvary = true;
    }
    
    if(!solidComponentsValueRight){
      var solidCheck = document.getElementById("solidCheckRight");
      solidCheck.textContent = "";
      var iconElement = document.createElement("i");
      iconElement.className = "bi bi-exclamation-diamond-fill";
      solidCheck.appendChild(iconElement);
      var newParagraph = document.createElement("p");
      newParagraph.textContent = " Not null";
      newParagraph.style.color ="#D9534F";
      solidCheck.appendChild(newParagraph);
      checkRightOvary = false;
      if(cout == 0){
        document.querySelector('input[name="solid_componets_right"]').focus();
        ++cout;
        createToast('error', 'fa-solid fa-circle-exclamation', 'Error', 'Please complete all information');
      }
    }else{
      var solidCheck = document.getElementById("solidCheckRight");
      solidCheck.textContent = "";
      checkRightOvary = true;
    }
    
    if(!lesionDiameterValueRight){
      var lesionCheck = document.getElementById("lesionCheckRight");
      lesionCheck.textContent = "";
      var iconElement = document.createElement("i");
      iconElement.className = "bi bi-exclamation-diamond-fill";
      lesionCheck.appendChild(iconElement);
      var newParagraph = document.createElement("p");
      newParagraph.textContent = " Not null";
      newParagraph.style.color ="#D9534F";
      lesionCheck.appendChild(newParagraph);
      checkRightOvary = false;
      if(cout == 0){
        document.querySelector('input[name="lesionDiameter"]').focus();
        ++cout;
        createToast('error', 'fa-solid fa-circle-exclamation', 'Error', 'Please complete all information');
      }
    }else{
      var lesionCheck = document.getElementById("lesionCheckRight");
      lesionCheck.textContent = "";
      checkRightOvary = true;
    }
    
    if(checkRightOvary == true){
    resultsDiagnosis.style.display = "block";
    
    for (var i = 0; i < resultRight.length; i++) {
      resultRight[i].style.display = "block";
    }
    
    var z1 = -6.7468 + 0.0326 * agePatientValueRight + 1.5513 * ascitesSelectValueRight + 1.1737 * bloodFlowSelectValueRight + 0.0496 * solidComponentsValueRight + 1.1421 * cystWallsValueRight - 2.3550 * acousticShadowsValueRight + 1.5985 * historyOvarianValueRight - 0.9983 * hormonalTherapyValueRight + 0.00841 * lesionDiameterValueRight- 0.8577 * painDuringExaminationValueRight + 0.9281 * purelySolidTumorValueRight + 0.4916 * colorScoreValueRight;
    var LR1 = 1 / (1 + Math.exp(-z1));
    var z2 = -5.3718 + 0.0354*agePatientValueRight + 1.6159*ascitesSelectValueRight + 1.1768*bloodFlowSelectValueRight + 0.0697*solidComponentsValueRight + 0.9586*cystWallsValueRight - 2.9486*acousticShadowsValueRight;
    var LR2 = 1 / (1 + Math.exp(-z2));
    
    var LR1RightOvary = document.getElementById("LR1_Result_RightOvary");
    LR1RightOvary.innerHTML = (LR1*100).toFixed(2);
    
    var LR2RightOvary = document.getElementById("LR2_Result_RightOvary");
    LR2RightOvary.innerHTML = (LR2*100).toFixed(2);
    
    google.charts.load('current', {'packages':['corechart']});
    google.charts.setOnLoadCallback(drawChart_LR1Right);
    
        function drawChart_LR1Right() {
    
          var data = google.visualization.arrayToDataTable([
            ['Task', 'Hours per Day'],
            ['Benign (%)', 1 - LR1],
            ['Manligant (%)', LR1]
          ]);
    
          var options = {
            title: 'LR1'
          };
    
          var chart = new google.visualization.PieChart(document.getElementById('piechart_LR1Right'));
    
          chart.draw(data, options);
        }
        
    google.charts.load('current', {'packages':['corechart']});
    google.charts.setOnLoadCallback(drawChart_LR2Right);
    
        function drawChart_LR2Right() {
    
          var data = google.visualization.arrayToDataTable([
            ['Task', 'Hours per Day'],
            ['Benign (%)', 1 - LR2],
            ['Manligant (%)', LR2]
          ]);
    
          var options = {
            title: 'LR2'
          };
    
          var chart = new google.visualization.PieChart(document.getElementById('piechart_LR2Right'));
    
          chart.draw(data, options);
        }
      }
    }
    if(checkboxL.checked && checkboxR.checked && checkLeftOvary && checkRightOvary){
      createToast('success', 'fa-solid fa-circle-check', 'Success', 'Please see the results below');
      document.getElementById('results-diagnosis').scrollIntoView({ behavior: 'smooth' });
    }else if(checkboxL.checked && !checkboxR.checked && checkLeftOvary){
      createToast('success', 'fa-solid fa-circle-check', 'Success', 'Please see the results below');
      document.getElementById('results-diagnosis').scrollIntoView({ behavior: 'smooth' });
    }else if(checkboxR.checked && !checkboxL.checked && checkRightOvary){
      createToast('success', 'fa-solid fa-circle-check', 'Success', 'Please see the results below');
      document.getElementById('results-diagnosis').scrollIntoView({ behavior: 'smooth' });
    }
    if(!checkboxL.checked && !checkboxR.checked){
      createToast('warning', 'fa-solid fa-triangle-exclamation', 'Warning', 'No information was entered');
    }

}
