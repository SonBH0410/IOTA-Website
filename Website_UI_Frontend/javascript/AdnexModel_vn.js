var cout = 0;
var checkLeftOvary = true;
var checkRightOvary = true;

var resultsDiagnosis = document.getElementById("results-diagnosis");
resultsDiagnosis.style.display = "none";
var resultLeft = document.getElementsByClassName('text-alert-left');
var resultRight = document.getElementsByClassName('text-alert-right');

for (var i = 0; i < resultLeft.length; i++) {
  resultLeft[i].style.display = "none";
}

for (var i = 0; i < resultRight.length; i++) {
  resultRight[i].style.display = "none";
}

function resetCheckboxes_LR() {
  window.location.href = window.location.href;
}

function checkResults_LR(){
  cout = 0;
    var checkboxL  = document.querySelector('input[name="checkbox-left-ovary"]');
    if(checkboxL.checked){
      //==================================================================================
    var age = parseInt(document.querySelector('input[name="age_patient_left"]').value);
    var center = parseInt(document.getElementById('oncologyCenterLeft').value);
    var lesion = parseInt(document.querySelector('input[name="maximalDiameterLeft"]').value);
    var solidComponent = parseInt(document.querySelector('input[name="solid_part_left"]').value);
    var cystLocules = parseInt(document.getElementById('cystLoculesLeft').value);
    var structures = parseInt(document.getElementById('papillationsLeft').value);
    var acousticShadows = parseInt(document.getElementById('acousticShadowsLeft').value);
    var ascites = parseInt(document.getElementById('ascitesPresentLeft').value);
    var CA125 = parseInt(document.querySelector('input[name="ca125Left"]').value); 
    
    if(!age){
      var ageCheck = document.getElementById("ageCheck");
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
        createToast('error', 'fa-solid fa-circle-exclamation', 'Chú ý', 'Vui lòng điền đầy đủ thông tin');
      }
    }else{
      var ageCheck = document.getElementById("ageCheck");
      ageCheck.textContent = "";
    }

    if(!lesion){
      var lesionCheck = document.getElementById("lesionCheck");
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
        document.querySelector('input[name="maximalDiameterLeft"]').focus();
        ++cout;
        createToast('error', 'fa-solid fa-circle-exclamation', 'Chú ý', 'Vui lòng điền đầy đủ thông tin');
      }
    }else{
      var lesionCheck = document.getElementById("lesionCheck");
      lesionCheck.textContent = "";
    }

    if(!solidComponent){
      var solidCheck = document.getElementById("solidCheck");
      solidCheck.textContent = "";
      var iconElement = document.createElement("i");
      iconElement.className = "bi bi-exclamation-diamond-fill";
      solidCheck.appendChild(iconElement);
      var newParagraph = document.createElement("p");
      newParagraph.textContent = " Not null";
      newParagraph.style.display = "inline";
      newParagraph.style.color ="#D9534F";
      solidCheck.appendChild(newParagraph);
      checkLeftOvary = false;
      if(cout == 0){
        ++cout;
        document.querySelector('input[name="solid_part_left"]').focus();
        createToast('error', 'fa-solid fa-circle-exclamation', 'Chú ý', 'Vui lòng điền đầy đủ thông tin');
      }
    }else{
      var solidCheck = document.getElementById("solidCheck");
      solidCheck.textContent = "";
    }

    if(!CA125){
      var ca125Check = document.getElementById("ca125Check");
      ca125Check.textContent = "";
      var iconElement = document.createElement("i");
      iconElement.className = "bi bi-exclamation-diamond-fill";
      ca125Check.appendChild(iconElement);
      var newParagraph = document.createElement("p");
      newParagraph.textContent = " Not null";
      newParagraph.style.color ="#D9534F";
      ca125Check.appendChild(newParagraph);
      checkLeftOvary = false;
      if(cout == 0){
        ++cout;
        document.querySelector('input[name="ca125Left"]').focus();
        createToast('error', 'fa-solid fa-circle-exclamation', 'Chú ý', 'Vui lòng điền đầy đủ thông tin');
      }
    }else{
      var ca125Check = document.getElementById("ca125Check");
      ca125Check.textContent = "";
    }

    if(age && solidComponent && lesion && CA125){
      checkLeftOvary = true;
    }

    if(checkLeftOvary == true){
      resultsDiagnosis.style.display = "block";

    for (var i = 0; i < resultLeft.length; i++) {
      resultLeft[i].style.display = "block";
    }

    var z1 = -7.577663 + 0.004506 * age + 0.111642 * Math.log2(CA125) + 0.372046 * Math.log2(lesion) + 6.967853 * (solidComponent / lesion) - 5.65588 * Math.pow(solidComponent / lesion, 2) + 1.375079 * cystLocules + 0.604238 * structures - 2.04157 * acousticShadows + 0.971061 * ascites + 0.953043 * center;

    var z2 = -12.276041 + 0.017260 * age + 0.197249 * Math.log2(CA125) + 0.873530 * Math.log2(lesion) + 9.583053 * (solidComponent / lesion) - 5.83319 * Math.pow(solidComponent / lesion, 2) + 0.791873 * cystLocules + 0.400369 * structures - 1.87763 * acousticShadows + 0.452731 * ascites + 0.452484 * center;
    
    var z3 = -14.915830 + 0.051239 * age + 0.765456 * Math.log2(CA125) + 0.430477 * Math.log2(lesion) + 10.37696 * (solidComponent / lesion) - 5.70975 * Math.pow(solidComponent / lesion, 2) + 0.273692 * cystLocules + 0.389874 * structures - 2.35516 * acousticShadows + 1.348408 * ascites + 0.459021 * center;
    
    var z4 = -11.909267 + 0.033601 * age + 0.276166 * Math.log2(CA125) + 0.449025 * Math.log2(lesion) + 6.644939 * (solidComponent / lesion) - 2.30330 * Math.pow(solidComponent / lesion, 2) + 0.899980 * cystLocules + 0.215645 * structures - 2.49845 * acousticShadows + 1.636407 * ascites + 0.808887 * center;
    
    var P_Benign = ((1 / (1 + Math.exp(z1) + Math.exp(z2) + Math.exp(z3) + Math.exp(z4)))*100).toFixed(1);
    
    var P_Borderline = ((Math.exp(z1) / (1 + Math.exp(z1) + Math.exp(z2) + Math.exp(z3) + Math.exp(z4)))*100).toFixed(1);
    
    var P_stageICancer = ((Math.exp(z2) / (1 + Math.exp(z1) + Math.exp(z2) + Math.exp(z3) + Math.exp(z4)))*100).toFixed(1);
    
    var P_stageII_IVCancer = ((Math.exp(z3) / (1 + Math.exp(z1) + Math.exp(z2) + Math.exp(z3) + Math.exp(z4)))*100).toFixed(1);
    
    var P_metastaticCancer = ((Math.exp(z4) / (1 + Math.exp(z1) + Math.exp(z2) + Math.exp(z3) + Math.exp(z4)))*100).toFixed(1);

    //Show results out by test notification
    var benign = document.getElementById("benignTumor");
    var malignant = document.getElementById("malignant");
    var borderline = document.getElementById("borderline");
    var stageI = document.getElementById("stageI");
    var stageIIIV = document.getElementById("stageIIIV");
    var metastatic = document.getElementById("metastatic");

    benign.innerHTML = P_Benign ;
    malignant.innerHTML = (100 - P_Benign).toFixed(1);
    borderline.innerHTML = P_Borderline;
    stageI.innerHTML = P_stageICancer;
    stageIIIV.innerHTML =  P_stageII_IVCancer;
    metastatic.innerHTML = P_metastaticCancer;

    // Draw stacked chart by Google chart

    google.charts.load('current', {'packages':['bar']});
    google.charts.setOnLoadCallback(drawChartLeft);

    function drawChartLeft() {
      var data = google.visualization.arrayToDataTable([
        ['Biểu đồ so sánh nguy cơ nền và nguy cơ của người bệnh','Lành tính','Ranh giới','Giai đoạn I','Giai đoạn II-IV','Di căn'],
        ['Nguy cơ nền', 68.1, 6.3, 7.5, 14.1, 4.0],
        ['Nguy cơ của người bệnh', P_Benign, P_Borderline, P_stageICancer, P_stageII_IVCancer, (100 - P_Benign).toFixed(1) - P_Borderline - P_stageICancer - P_stageII_IVCancer]
      ]);

      var numRows = data.getNumberOfRows();
      var numCols = data.getNumberOfColumns();
      for (var i = 2; i < numRows; i++) {
          var total = data.getValue(i, 0);
          for (var j = 1; j < numCols; j++) {
              var value = data.getValue(i, j);
              var percentage = (value / total) * 100;
              data.setValue(i, j, percentage > 100 ? 100 : percentage);
          }
      }
        var options = {
            width: 600,
            height: 400,
            legend: { position: 'top', maxLines: 3 },
            bar: { groupWidth: '50%' },
            isStacked: true,
            vAxis: { maxValue: 100 },
            series: {
                0: { color: '#26741E'},
                1: { color: '#B89C14'},
                2: { color: '#BF6801'},
                3: { color: '#BF3A01'},
                4: { color: '#811200'}
            },
        };

        var chart = new google.charts.Bar(document.getElementById('columnchart_material'));

        chart.draw(data, google.charts.Bar.convertOptions(options));
      }
    }
  }
  
  //==================================================================================
  var checkboxR  = document.querySelector('input[name="checkbox-right-ovary"]');
  if(checkboxR.checked){
    var ageR = parseInt(document.querySelector('input[name="age_patient_right"]').value);
    var centerR = parseInt(document.getElementById('oncologyCenterRight').value);
    var lesionR = parseInt(document.querySelector('input[name="maximalDiameterRight"]').value);
    var solidComponentR = parseInt(document.querySelector('input[name="solid_part_right"]').value);
    var cystLoculesR = parseInt(document.getElementById('cystLoculesRight').value);
    var structuresR = parseInt(document.getElementById('papillationsRight').value);
    var acousticShadowsR = parseInt(document.getElementById('acousticShadowsRight').value);
    var ascitesR = parseInt(document.getElementById('ascitesPresentRight').value);
    var CA125R = parseInt(document.querySelector('input[name="ca125Right"]').value);

    if(!ageR){
      var ageCheckR = document.getElementById("ageCheckRight");
      ageCheckR.textContent = "";
      var iconElement = document.createElement("i");
      iconElement.className = "bi bi-exclamation-diamond-fill";
      ageCheckR.appendChild(iconElement);
      var newParagraph = document.createElement("p");
      newParagraph.textContent = " Not null";
      newParagraph.style.color ="#D9534F";
      ageCheckR.appendChild(newParagraph);
      checkRightOvary = false;
      if(cout == 0){
        ++cout;
        document.querySelector('input[name="age_patient_right"]').focus();
        createToast('error', 'fa-solid fa-circle-exclamation', 'Chú ý', 'Vui lòng điền đầy đủ thông tin');
      }
    }else{
      var ageCheckR = document.getElementById("ageCheckRight");
      ageCheckR.textContent = "";
    }

    if(!lesionR){
      var lesionCheckR = document.getElementById("lesionCheckRight");
      lesionCheckR.textContent = "";
      var iconElement = document.createElement("i");
      iconElement.className = "bi bi-exclamation-diamond-fill";
      lesionCheckR.appendChild(iconElement);
      var newParagraph = document.createElement("p");
      newParagraph.textContent = " Not null";
      newParagraph.style.color ="#D9534F";
      lesionCheckR.appendChild(newParagraph);
      checkRightOvary = false;
      if(cout == 0){
        ++cout;
        document.querySelector('input[name="maximalDiameterRight"]').focus();
        createToast('error', 'fa-solid fa-circle-exclamation', 'Chú ý', 'Vui lòng điền đầy đủ thông tin');
      }
    }else{
      var lesionCheckR = document.getElementById("lesionCheckRight");
      lesionCheckR.textContent = "";
    }

    if(!solidComponentR){
      var solidCheckR = document.getElementById("solidCheckRight");
      solidCheckR.textContent = "";
      var iconElement = document.createElement("i");
      iconElement.className = "bi bi-exclamation-diamond-fill";
      solidCheckR.appendChild(iconElement);
      var newParagraph = document.createElement("p");
      newParagraph.textContent = " Not null";
      newParagraph.style.color ="#D9534F";
      solidCheckR.appendChild(newParagraph);
      checkRightOvary = false;
      if(cout == 0){
        ++cout;
        document.querySelector('input[name="solid_part_right"]').focus();
        createToast('error', 'fa-solid fa-circle-exclamation', 'Chú ý', 'Vui lòng điền đầy đủ thông tin');
      }
    }else{
      var solidCheckR = document.getElementById("solidCheckRight");
      solidCheckR.textContent = "";
    }

    if(!CA125R){
      var ca125CheckR = document.getElementById("ca125CheckRight");
      ca125CheckR.textContent = "";
      var iconElement = document.createElement("i");
      iconElement.className = "bi bi-exclamation-diamond-fill";
      ca125CheckR.appendChild(iconElement);
      var newParagraph = document.createElement("p");
      newParagraph.textContent = " Not null";
      newParagraph.style.color ="#D9534F";
      ca125CheckR.appendChild(newParagraph);
      checkRightOvary = false;
      if(cout == 0){
        ++cout;
        document.querySelector('input[name="ca125Right"]').focus();
        createToast('error', 'fa-solid fa-circle-exclamation', 'Chú ý', 'Vui lòng điền đầy đủ thông tin');
      }
    }else{
      var ca125CheckR = document.getElementById("ca125CheckRight");
      ca125CheckR.textContent = "";
    }

    if(CA125R && lesionR && solidComponentR && ageR){
      checkRightOvary=true;
    }
    if(checkRightOvary == true){
      resultsDiagnosis.style.display = "block";
      for (var i = 0; i < resultRight.length; i++) {
        resultRight[i].style.display = "block";
      }
  
      var z1R = -7.577663 + 0.004506 * ageR + 0.111642 * Math.log2(CA125R) + 0.372046 * Math.log2(lesionR) + 6.967853 * (solidComponentR / lesionR) - 5.65588 * Math.pow(solidComponentR / lesionR, 2) + 1.375079 * cystLoculesR + 0.604238 * structuresR - 2.04157 * acousticShadowsR + 0.971061 * ascitesR + 0.953043 * centerR;
  
      var z2R = -12.276041 + 0.017260 * ageR + 0.197249 * Math.log2(CA125R) + 0.873530 * Math.log2(lesionR) + 9.583053 * (solidComponentR / lesionR) - 5.83319 * Math.pow(solidComponentR / lesionR, 2) + 0.791873 * cystLoculesR + 0.400369 * structuresR - 1.87763 * acousticShadowsR + 0.452731 * ascitesR + 0.452484 * centerR;
      
      var z3R = -14.915830 + 0.051239 * ageR + 0.765456 * Math.log2(CA125R) + 0.430477 * Math.log2(lesionR) + 10.37696 * (solidComponentR / lesionR) - 5.70975 * Math.pow(solidComponentR / lesionR, 2) + 0.273692 * cystLoculesR + 0.389874 * structuresR - 2.35516 * acousticShadowsR + 1.348408 * ascitesR + 0.459021 * centerR;
      
      var z4R = -11.909267 + 0.033601 * ageR + 0.276166 * Math.log2(CA125R) + 0.449025 * Math.log2(lesionR) + 6.644939 * (solidComponentR / lesionR) - 2.30330 * Math.pow(solidComponentR / lesionR, 2) + 0.899980 * cystLoculesR + 0.215645 * structuresR - 2.49845 * acousticShadowsR + 1.636407 * ascitesR + 0.808887 * centerR;
      
      var P_BenignR = ((1 / (1 + Math.exp(z1R) + Math.exp(z2R) + Math.exp(z3R) + Math.exp(z4R))) * 100).toFixed(1);
      
      var P_BorderlineR = ((Math.exp(z1R) / (1 + Math.exp(z1R) + Math.exp(z2R) + Math.exp(z3R) + Math.exp(z4R))) * 100).toFixed(1);
      
      var P_stageICancerR = ((Math.exp(z2R) / (1 + Math.exp(z1R) + Math.exp(z2R) + Math.exp(z3R) + Math.exp(z4R))) * 100).toFixed(1);
      
      var P_stageII_IVCancerR = ((Math.exp(z3R) / (1 + Math.exp(z1R) + Math.exp(z2R) + Math.exp(z3R) + Math.exp(z4R))) * 100).toFixed(1);
      
      var P_metastaticCancerR = ((Math.exp(z4R) / (1 + Math.exp(z1R) + Math.exp(z2R) + Math.exp(z3R) + Math.exp(z4R))) * 100).toFixed(1);
      
      //Show results out by test notification
      var benignR = document.getElementById("benignTumorRight");
      var malignantR = document.getElementById("malignantRight");
      var borderlineR = document.getElementById("borderlineRight");
      var stageIR = document.getElementById("stageIRight");
      var stageIIIVR = document.getElementById("stageIIIVRight");
      var metastaticR = document.getElementById("metastaticRight");
  
      benignR.innerHTML = P_BenignR ;
      malignantR.innerHTML = (100 - P_BenignR).toFixed(1);
      borderlineR.innerHTML = P_BorderlineR;
      stageIR.innerHTML = P_stageICancerR;
      stageIIIVR.innerHTML =  P_stageII_IVCancerR;
      metastaticR.innerHTML = P_metastaticCancerR;
  
      google.charts.load('current', {'packages':['bar']});
      google.charts.setOnLoadCallback(drawChartRight);
  
      function drawChartRight() {
        var data = google.visualization.arrayToDataTable([
          ['Biểu đồ so sánh nguy cơ nền và nguy cơ của người bệnh','Lành tính','Ranh giới','Giai đoạn I','Giai đoạn II-IV','Di căn'],
          ['Nguy cơ nền', 68.1, 6.3, 7.5, 14.1, 4.0],
          ['Nguy cơ của người bệnh', P_BenignR, P_BorderlineR, P_stageICancerR, P_stageII_IVCancerR, (100 - P_BenignR).toFixed(1) - P_BorderlineR - P_stageICancerR - P_stageII_IVCancerR]
        ]);
  
        var numRows = data.getNumberOfRows();
        var numCols = data.getNumberOfColumns();
        for (var i = 2; i < numRows; i++) {
            var total = data.getValue(i, 0);
            for (var j = 1; j < numCols; j++) {
                var value = data.getValue(i, j);
                var percentage = (value / total) * 100;
                data.setValue(i, j, percentage > 100 ? 100 : percentage);
            }
        }
          var options = {
              width: 600,
              height: 400,
              legend: { position: 'top', maxLines: 3 },
              bar: { groupWidth: '50%' },
              isStacked: true,
              vAxis: { maxValue: 100 },
              series: {
                  0: { color: '#26741E'},
                  1: { color: '#B89C14'},
                  2: { color: '#BF6801'},
                  3: { color: '#BF3A01'},
                  4: { color: '#811200'}
              },
              annotations: {
                textStyle: {
                    fontSize: 11,
                    color: 'black',
                },
                alwaysOutside: true,
            },
          };
  
          var chart = new google.charts.Bar(document.getElementById('columnchart_materialRight'));
  
          chart.draw(data, google.charts.Bar.convertOptions(options));
        }
      } 
    }
    if(checkboxL.checked && checkboxR.checked && checkLeftOvary && checkRightOvary){
      createToast('success', 'fa-solid fa-circle-check', 'Thành công', 'Vui lòng xem kết quả bên dưới');
      document.getElementById('results-diagnosis').scrollIntoView({ behavior: 'smooth' });
    }else if(checkboxL.checked && !checkboxR.checked && checkLeftOvary){
      createToast('success', 'fa-solid fa-circle-check', 'Thành công', 'Vui lòng xem kết quả bên dưới');
      document.getElementById('results-diagnosis').scrollIntoView({ behavior: 'smooth' });
    }else if(checkboxR.checked && !checkboxL.checked && checkRightOvary){
      createToast('success', 'fa-solid fa-circle-check', 'Thành công', 'Vui lòng xem kết quả bên dưới');
      document.getElementById('results-diagnosis').scrollIntoView({ behavior: 'smooth' });
    }

    if(!checkboxL.checked && !checkboxR.checked){
      createToast('warning', 'fa-solid fa-triangle-exclamation', 'Cảnh báo', 'Không có thông tin được điền');
    }
}