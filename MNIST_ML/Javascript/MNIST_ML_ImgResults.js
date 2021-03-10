function ImgDisplayApp ()
{
    var selectedNum = 0;

    // Instead of loading and displaying the data from the CSV, this was done to keep things simple.
    var data = [ { numTruePos:  941, numTrueNeg: 8953, numFalsePos:  67, numFalseNeg:  39 },
                 { numTruePos: 1117, numTrueNeg: 8786, numFalsePos:  79, numFalseNeg:  18 },
                 { numTruePos:  757, numTrueNeg: 8919, numFalsePos:  49, numFalseNeg: 275 },
                 { numTruePos:  557, numTrueNeg: 8978, numFalsePos:  12, numFalseNeg: 453 },
                 { numTruePos:  927, numTrueNeg: 8824, numFalsePos: 194, numFalseNeg:  55 },
                 { numTruePos:  785, numTrueNeg: 8707, numFalsePos: 401, numFalseNeg: 107 },
                 { numTruePos:  686, numTrueNeg: 9029, numFalsePos:  13, numFalseNeg: 272 },
                 { numTruePos:  921, numTrueNeg: 8902, numFalsePos:  70, numFalseNeg: 107 },
                 { numTruePos:  605, numTrueNeg: 8786, numFalsePos: 240, numFalseNeg: 369 },
                 { numTruePos:  587, numTrueNeg: 8889, numFalsePos: 102, numFalseNeg: 422 } ];


    //
    // Callback to handle the updates...
    //
    function numUpdate () 
    {
        var val = +this.getAttribute ('val');
        if (val != selectedNum)
        {
            // update the highlighted element...
            document.getElementById ('Num_' + selectedNum).classList.remove ('cellSel');
            document.getElementById ('Num_' + val).classList.add ('cellSel');
                      
            changeImages (val);

            selectedNum = val;
        }
    };


    //
    // Change the number and text...
    //
    function changeImages (number)
    {
        document.getElementById ('TNImg').src = 'MNIST_ML_ImgResults/BinaryClassifier_SGD_' + number + '_TN.jpg';
        document.getElementById ('FPImg').src = 'MNIST_ML_ImgResults/BinaryClassifier_SGD_' + number + '_FP.jpg';
        document.getElementById ('FNImg').src = 'MNIST_ML_ImgResults/BinaryClassifier_SGD_' + number + '_FN.jpg';
        document.getElementById ('TPImg').src = 'MNIST_ML_ImgResults/BinaryClassifier_SGD_' + number + '_TP.jpg';

        document.getElementById ('TNValue').value = data [number].numTrueNeg;
        document.getElementById ('FPValue').value = data [number].numFalsePos;
        document.getElementById ('FNValue').value = data [number].numFalseNeg;
        document.getElementById ('TPValue').value = data [number].numTruePos;

        document.getElementById ('TotalValue').value = data [number].numTrueNeg + data [number].numFalsePos + 
                                                       data [number].numFalseNeg + data [number].numTruePos;
    }


    // add callbacks for all the number buttons...
    var elements = document.getElementsByClassName ('cell');
    for (var i = 0; i < elements.length; i++) 
    {
        elements[i].addEventListener ('click', numUpdate, false);
    }

    // init the numbers and values
    changeImages (0);
}


