<html>

  <head>
    <!-- <meta http-equiv="Content-Type" content="text/html; charset=utf-8"> -->
    <meta http-equiv="content-type" content="text/html; charset=big5">
    <title>Plants Classification</title>
    <link rel=stylesheet type="text/css" href="style.css">
    
  
  </head>
  
  <body>
    <br>
    <header>
      <h1>Plants Classification: <br> via Xception net Transfer Learning</h1>
    
    </header>
    
     
    
    <div id="step1" class="main_scene">
    <span style="font-size:16pt;font-family:baby_blocks; ">Please upload a plant image.</span><br><hr>
    



    <?php
      // $flagFile = 'temp/flag.txt';
      // if(file_exists($flagFile)){
      //   chmod($flagFile, 0777);
      //   unlink($flagFile);
      // }
    
      // $wavfile = glob('demoSong/*.*'); 
      // echo "<li>Choose one way to select music :<br><br>";
      echo "<form action='' method='post' name='demo' enctype='multipart/form-data'> ";
      // echo "<input type='radio' id='useExampleMusic' name='musicType' value='demoSong' checked> ";
      // echo "<select name='demoSongList' size='1'>";
      
      // for($i=0; $i<=count($wavfile); $i++){
      //   if($i!=count($wavfile)){
      //     $wavfile[$i] = iconv("BIG5", "UTF-8",$wavfile[$i]); 
      //     echo "<option value='$wavfile[$i]'>".substr($wavfile[$i], 9, -4)."</option>";
      //   }
      //   else{
      //     echo "<option value='tip' selected>Select an example music clip </option>";
      //   }
        
      // }
      // echo "</select></li><br>";

      // echo "<input type='radio' id='useExampleMusic2' name='musicType' value='uploadSong' > "; 
      echo "Upload Your File: "; 
      echo "<input type='file' id='uploadMusic' name='upload' value='Upload Music' /><br><br>"; 
      echo "<center><input type='submit' value='Start Plants Recognition' /></center>";
      echo "</form>";
      
    ?>

    <?php

      $py_exe_path = 'C:\\Program Files\\Python36\\python.exe';
      $py_src_path = 'D:\\xampp\\htdocs\\demo\\plantsClassification\\PlantsPrediction.py';
      


      if(@$_FILES["upload"]["name"]!= NULL){
        $date_time = date("YmdHis");
        $filename = $date_time . "_";
        $uploadImage = "PlantsImage";
        if($_FILES["upload"]["type"]=="image/png"){
          move_uploaded_file($_FILES['upload']['tmp_name'], "PlantsImage/".$filename.".png");
          $uploadImage = "PlantsImage/".$filename.".png";
          //  echo "======filename is:upload/" . $filename . ".wav======";
        }
        elseif($_FILES["upload"]["type"]=="image/jpg"||$_FILES["upload"]["type"]=="image/jpeg"){
          move_uploaded_file($_FILES['upload']['tmp_name'], "PlantsImage/".$filename.".jpg");
          $uploadImage = "PlantsImage/".$filename.".jpg";
          //  echo "======filename is:upload/" . $filename . ".wav======";
        }

        $uploadImagePath = "PlantsImage/".$filename.".png";

        if($_FILES["upload"]["type"]=="image/jpg"||$_FILES["upload"]["type"]=="image/jpeg"){       
          $uploadImagePath = "PlantsImage/".$filename.".jpg";
        }
        // echo "Saving image is fine...";
        $exec_command = "\"{$py_exe_path}\" \"{$py_src_path}\" \"{$uploadImagePath}\" 2>&1";
        $read = exec($exec_command, $output, $return_var);
        // print_r($output);

        echo "Your picture:";
        echo "<br><br>";
        echo "<img src='$uploadImagePath' width=227px height=227px>";
        echo "<br><br>";

        echo "This plant is very likely to be:";
        echo "<br><br>";
        
        $ImagePath_1 = $output[11];
        $ImagePath_2 = $output[12];
        $ImagePath_3 = $output[13];
        $ImagePath_4 = $output[14];
        $ImagePath_5 = $output[15];
        echo mb_convert_encoding($output[6],"utf-8","big5");
        echo "<br><br>";
        echo "<img src='$ImagePath_1'>";
        echo "<br><br>";
        echo mb_convert_encoding($output[7],"utf-8","big5");
        echo "<br><br>";
        echo "<img src='$ImagePath_2'>";
        echo "<br><br>";
        echo mb_convert_encoding($output[8],"utf-8","big5");
        echo "<br><br>";
        echo "<img src='$ImagePath_3'>";
        echo "<br><br>";
        echo mb_convert_encoding($output[9],"utf-8","big5");
        echo "<br><br>";
        echo "<img src='$ImagePath_4'>";
        echo "<br><br>";
        echo mb_convert_encoding($output[10],"utf-8","big5");
        echo "<br><br>";
        echo "<img src='$ImagePath_5'>";
        echo "<br><br>";
        // echo mb_convert_encoding($read,"utf-8","big5");
      }

      // $uploadImagePath = $uploadImage;




    ?>


    
    <?php
      
      // $py_exe_path = 'C:\\Program Files\\Python35\\python.exe';
      // $py_src_path = 'D:\\xampp\\htdocs\\demo\\SVSGAN\\svsganTest.py';
      
      // if(@$_POST['musicType']=='demoSong' && $_POST['demoSongList']!='tip'){  
        
      //   $demoSong_big5 = iconv("UTF-8", "BIG5", $_POST['demoSongList']);
      //   $demoSong = $_POST['demoSongList'];
      //   $demoSong_wav='demoSong_wav/'.substr($demoSong_big5, 9, -4).'.wav';
      //   $demoNoise = 'pred_accomp_wav/'.substr($demoSong, 9, -4).'_accomp.wav';
      //   $demoSignal = 'pred_vocal_wav/'.substr($demoSong, 9, -4).'_vocal.wav';
        
      //   exec(" matlab -wait -nojvm -nosplash -nodisplay -nodesktop -r \" audio2wav('$demoSong_big5', 'demoSong_wav'); exit; \"");
      //   $exec_command = "\"{$py_exe_path}\" \"{$py_src_path}\" -f \"{$demoSong_wav}\"";
      //   exec($exec_command, $output, $return_var);
      //   //print_r($output);
        
      //   //echo $demoSong_big5;
      //   //print_r($return_var);
      //   //echo $demoSong_wav;
      //   //die('');

      //   echo  "<br><center><table border='1' style='width:60%'>".
      //         "<tr>".
      //           "<td><b><center><font color='blue'>Original Song:</font></center></td>".
      //           "<td><center><audio controls><source src='$demoSong'></audio></center></td>".
      //         "</tr>".
      //         "<tr>".
      //           "<td><b><center><font color='blue'>Separated Voice:</font></center></td>".
      //           "<td><center><audio controls><source src='$demoSignal'></audio></center></td>".
      //         "</tr>".
      //         "<tr>".
      //           "<td><b><center><font color='blue'>Separated Music:</font></center></td>".
      //           "<td><center><audio controls><source src='$demoNoise'></audio></center></td>".
      //         "</tr>".
      //       "</table></center><br>";

      // }
      // else if(@$_POST['musicType']=='uploadSong' && @$_FILES["upload"]["name"]!= NULL){
      //   //$uploadName = iconv("UTF-8", "BIG5", $_FILES["upload"]["name"]);
      //   //$uploadName = $_FILES["upload"]["name"];
      //   //echo $_FILES["upload"]["type"];
        
      //   $date_time = date("YmdHis"); //¥[¤J·í«e®É¶¡¥H«K³B²z¦PÀÉ®×¤£¦P®É¶¡¤W¶Ç¤§±¡ªp
      //   $filename = $date_time . "_";
      //   if($_FILES["upload"]["type"]=="audio/wav"||$_FILES["upload"]["type"]=="audio/x-wav"){
      //     move_uploaded_file($_FILES['upload']['tmp_name'], "uploadSong/".$filename.".wav");
      //     $uploadSong = "uploadSong/".$filename.".wav";
      //   //  echo "======filename is:upload/" . $filename . ".wav======";
      //   }
      //   else if($_FILES["upload"]["type"]=="audio/mp3"){
      //     move_uploaded_file($_FILES['upload']['tmp_name'], "uploadSong/".$filename.".mp3");
      //     $uploadSong = "uploadSong/".$filename.".mp3";
      //   //  echo "======filename is:upload/" . $filename . ".mp3======";
      //   } 
        
        
      //   //move_uploaded_file($_FILES['upload']['tmp_name'], "uploadSong/".$uploadName);
        
      //   //$uploadSong_big5 = iconv("UTF-8", "BIG5", "uploadSong/".$_FILES["upload"]["name"]);
      //   //@$uploadSong = "uploadSong/".$_FILES["upload"]["name"];
      //   //@$uploadSong_wav='uploadSong_wav/'.substr($_FILES["upload"]["name"],0, -4).'.wav';
        
      //   @$uploadSong_wav='uploadSong_wav/'.$filename.'.wav';
      //   @$uploadSong_clipped_wav="uploadSong_clipped_wav/".$filename.'.wav';
        
      //   @$uploadNoise = 'pred_accomp_wav/'.$filename.'_accomp.wav';
      //   @$uploadSignal = 'pred_vocal_wav/'.$filename.'_vocal.wav';
        
      //   //echo $uploadSong_wav;
        
      //   exec(" matlab -wait -nojvm -nosplash -nodisplay -nodesktop -r \" audio2wav('$uploadSong','uploadSong_wav'); exit; \"");
      //   $exec_command = "\"{$py_exe_path}\" \"{$py_src_path}\" -f \"{$uploadSong_wav}\"";
      //   exec($exec_command, $output, $return_var);
        

        
      //   echo  "<br><center><table border='1' style='width:60%'>".
      //         "<tr>".
      //         "<td><b><center><font color='blue'>Original Song:</font></center></td>".
      //         "<td><center><audio controls><source src='$uploadSong_clipped_wav'></audio></center></td>".
      //         "</tr>".
      //         "<tr>".
      //           "<td><b><center><font color='blue'>Separated Voice:</font></center></td>".
      //           "<td><center><audio controls><source src='$uploadSignal'></audio></center></td>".
      //         "</tr>".
      //         "<tr>".
      //           "<td><b><center><font color='blue'>Separated Music:</font></center></td>".
      //           "<td><center><audio controls><source src='$uploadNoise'></audio></center></td>".
      //         "</tr>".
      //       "</table></center><br>";
      // }
    
     ?>

    
    </div>
    
    <table border="0" cellspacing="10" cellpadding="10" align="center" >
        
      <!--<tr>
        <td width="750px" > <font size="4"><strong>This demo is based on SVSGAN proposed in the following paper:</strong> </td> 
      </tr>-->

      <tr>
        <td width="750px" >
        <font><strong>This demo is based on the two following theses:</strong><br>
        <font>Chih-Heng Hsiao, Jyh-Shing R. Jang,
        "<strong><em><a href="http://mirlab.org/users/joshua.hsiao/thesis/thesis.pdf" target="_blank">Plant Image Recognition by Transfer Learning and Plant Organ Separated Model</a></em></strong>",<br>

        <font>Yi-Shuan Chen, Jyh-Shing R. Jang,
        "<strong><em><a href="http://mirlab.org/users/yihsuan.chen/paper/untitled.pdf" target="_blank">Plant Image Recognition with CNN and Re-classification</a></em></strong>"<br>
       </td> 
      </tr>
    <!--  <tr>
        <td width="650px" > <font size="4"><strong> Here are some separated <a href=svsgan_paper_result.html>results</a> of the paper.<strong></td> 
      </tr> -->
    </table>
    
    
    
    <footer>
      <p style="color:black;">
        <Strong>Maintained by <a href="http://mirlab.org/users/joshua.hsiao/" target="_blank">Chih-Heng Hsiao</a> (<a href="mailto:joshua.hsiao@mirlab.org" target="_top" style="color: black">joshua.hsiao@mirlab.org</a>)</strong>
      </p>

     <!--  <span class="recom_browser">Recommended Browsers:
        <img src="images/chrome_logo.png" width=3% height=5% title="Google Chrome"/>
        <img src="images/firefox_logo.png" width=3% height=5% title="FireFox"/>
        <img src="images/safari_logo.png" width=3% height=5% title="Safari"/>
      </span><br> -->
      <span class="copyright"><img src="images/mirlab_logo.png" width=133 height=60 title="MIRLAB"/></span>
  <!--  <span >※Recommended Browsers：<img src="images/chrome_logo.png" width=3% height=5% title="Google Chrome"/><img src="images/firefox_logo.png" width=3% height=5% title="Safari"/><img src="images/safari_logo.png" width=3% height=5% title="Safari"/></span><br> -->
  <!--  <span class="copyright"><img src="images/mirlab_logo.png" width=133 height=60 title="MIRLAB"/></span> -->
    </footer>
    <!-- <script type="text/javascript">
      document.getElementById('uploadMusic').addEventListener('click', function(event) {
        document.getElementById('useExampleMusic2').checked = true;
      });
    </script> -->
    
  </body>
</html>