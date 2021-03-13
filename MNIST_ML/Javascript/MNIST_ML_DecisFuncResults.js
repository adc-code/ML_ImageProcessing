function D3App ()
{
    // The data...
    var dataFile = 'FullClassifier_DecisionFuncResults.csv';

    var dataset, xScale, yScale, xAxis, yAxis;

    var selectedID = 'IB00';

    var barLabels = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ];

 
    // SVG Width, height, and some added spacing
    var margin = {
            top:    5,
            right:  5,
            bottom: 20,
            left:   45
    };

    var width  = 500 - margin.left - margin.right;
    var height = 300 - margin.top - margin.bottom;

    var updateDuration = 500;


    //
    // Function used to parse the CSV.  
    // 
    var rowConverter = function (d) 
    {
        console.log (d);

        return [
            parseInt (d['caseNum']),         //  0
            parseInt (d['expectedValue']),   //  1
            parseInt (d['predictedValue']),  //  2
            d['state'],                      //  3
            parseInt (d['index']),           //  4
            parseFloat (d['df0']),           //  5
            parseFloat (d['df1']),           //  6
            parseFloat (d['df2']),           //  7
            parseFloat (d['df3']),           //  8
            parseFloat (d['df4']),           //  9
            parseFloat (d['df5']),           // 10
            parseFloat (d['df6']),           // 11
            parseFloat (d['df7']),           // 12
            parseFloat (d['df8']),           // 13
            parseFloat (d['df9'])            // 14
        ];  
    }


    //
    // Read the CSV...
    //
    d3.csv (dataFile, rowConverter).then (function (data) 
    {
        // uncomment for testing
        // console.log (data);   
         
        dataset = data;

        var selectedCase  = 0; 
        var selectedNum   = 0; // 0 to 9
        var selectedState = 0; // 0 or 1

        UpdateLabels ();
        UpdateImages ();

        decisionFuncValues = dataset[0].slice (5);
        dfMin = d3.min (decisionFuncValues);
        dfMax = d3.max (decisionFuncValues);
        maxValue = d3.max ( [ Math.abs (dfMin), Math.abs (dfMax) ] )

        var svg = d3.select ('#graph').append('svg')
                    .attr ('width', width + margin.left + margin.right)
                    .attr ('height', height + margin.top + margin.bottom)
                    .append ('g')
                    .attr ('transform', 'translate(' + margin.left + ',' + margin.top + ')');

        // Define the scales to convert our data to screen coordinates
        xScale = d3.scaleLinear ()
                   .domain ( [ 0, 10 ])
                   .range ( [ 0, width ] ); 

        yScale = d3.scaleLinear ()
                   //.domain ( [ d3.min (decisionFuncValues), d3.max (decisionFuncValues) ] )
                   .domain ( [ -1 * maxValue, maxValue ] )
                   .range ( [ height, 0 ] ); 

        xAxis = d3.axisBottom ()
                  .scale (xScale)
                  .ticks (10)
                  //.tickSize ( -height )
                  .tickSizeInner ( 10 )
                  .tickSizeOuter ( 10 )
                  .tickPadding ( -4 )
                  .tickFormat ( function (d, i) { return barLabels[i]; });

        // Define Y axis
        yAxis = d3.axisLeft ()
                  .scale (yScale)
                  .ticks (10);

        // Create axes..
        svg.append ('g')
           .attr ('class', 'axis')
           .attr ('transform', 'translate(0,' + (height) + ')')
           .call (xAxis)
           .selectAll ('text')  
           .style ('text-anchor', 'middle')
           .attr ('dx', xScale(0.5) );
 
        svg.append ('g')
           .attr ('class', 'y axis')
           .call (yAxis); 

        var space    = xScale(1.025) - xScale(1);
        var barWidth = xScale(1.95) - xScale(1);

        for (var i = 0; i < 10; i++)
        {
            var xValue = 1.25*space + xScale(i);
            if (i == 0)
                xValue += 0.5*space;
 
            var yValue = yScale (0);
            if (dataset[0][5 + i] > 0)
                yValue = yScale (dataset[0][5 + i]);

            var barHeight = 1;
            if (dataset[0][5 + i] != 0)
                barHeight = Math.abs(yScale(0) - yScale(dataset[0][5 + i]));

            svg.append ('rect')
               .attr ('id',     ('rect_' + i) )
               .attr ('x',      xValue )
               .attr ('y',      yValue )
               .attr ('width',  barWidth )
               .attr ('height', barHeight )
               .attr ('fill',   '#6666cc' );
        }


        //
        // Used to redraw the graph
        //
        //var UpdateGraph = function (selData, selRange)
        function UpdateGraph ()
        {
            var offset = selectedCase * 20 + selectedNum + 10*selectedState;

            // console.log ('UpdateGraph...', offset);

            var decisionFuncValues = dataset[offset].slice (5);
            var dfMin = d3.min (decisionFuncValues);
            var dfMax = d3.max (decisionFuncValues);
            var maxValue = d3.max ( [ Math.abs (dfMin), Math.abs (dfMax) ] );

            // update the y-axis
            yScale.domain ( [ -1*maxValue, maxValue ] )

            svg.select ('.y.axis')
               .transition ()
               .duration (1000)
               .call (yAxis);

            for (var i = 0; i < 10; i++)
            {
                var yValue = yScale (0);
                if (dataset[offset][5 + i] > 0)
                    yValue = yScale (dataset[offset][5 + i]);

                var barHeight = 1;
                if (dataset[offset][5 + i] != 0)
                    barHeight = Math.abs(yScale(0) - yScale(dataset[offset][5 + i]));

                d3.select ('#rect_' + i)
                  .transition ()
                  .duration (updateDuration)
                  .attr ('y',      yValue )
                  .attr ('height', barHeight );
            }

        }  // update graph function


        //
        // UpdateImages: Update images when a case changes...
        //
        function UpdateImages ()
        {
            var dataOffset = 20*selectedCase;
            var idStr = 'imgOK_';
            console.log ('UpdateImages...', dataOffset);

            for (var i = 0; i < 10; i++)
            {
                var imgFileName = 'FullClassifier_DecisionFuncResults/Case_' + selectedCase + '_' 
                        + dataset[dataOffset + i][1] + '_' + dataset[dataOffset + i][2] + '_OK_' 
                        + dataset [dataOffset + i][4] + '.jpg';

                var imgElem = document.getElementById (idStr + i);
                imgElem.src = imgFileName;
            }
            
            dataOffset = 20*selectedCase + 10;
            idStr = 'imgKO_';

            for (var i = 0; i < 10; i++)
            {
                var imgFileName = 'FullClassifier_DecisionFuncResults/Case_' + selectedCase + '_' 
                        + dataset[dataOffset + i][1] + '_' + dataset[dataOffset + i][2] + '_KO_' 
                        + dataset [dataOffset + i][4] + '.jpg';

                var imgElem = document.getElementById (idStr + i);
                imgElem.src = imgFileName;
            }
        }


        //
        // UpdateLabels: Used to update the expected/predicted text messages....
        // 
        function UpdateLabels ()
        {
            var dataOffset = 20*selectedCase;

            var corrTextElem = document.getElementById ('corrText');
            corrTextElem.innerHTML = '<i>' + 'Expected '  + dataset[dataOffset][1] + '... ' 
                                           + 'Predicted ' + dataset[dataOffset][1] + '... ' + '</i>' 
                                           + '&#x2714';

            var inCorrTextElem = document.getElementById ('incorrText');
            dataOffset += 10;
            inCorrTextElem.innerHTML = '<i>' + 'Expected '  + dataset[dataOffset][1] + '... ' 
                                             + 'Predicted ' + dataset[dataOffset][2] + '... ' + '</i>'
                                             + '&#x2718';
        }


        //
        // Callback to handle clicking on an imgBox
        //
        d3.selectAll ('.imgBox').on ('click', function ()
        {
            var currID = d3.select (this).node().getAttribute ('id');

            if (currID != selectedID)
            {
                var selImg = +d3.select (this).node().getAttribute ('sel');

                selectedNum   = selImg % 10;
                selectedState = Math.floor (selImg / 10);

                d3.select ('#' + selectedID).node().classList.remove ('selImgBox');
                d3.select ('#' + currID).node().classList.add ('selImgBox');
 
                // uncomment for testing...
                // console.log ('on click... ', selImg, selectedNum, selectedState);

                selectedID = currID;

                UpdateGraph ();
            }

        } );

      
        //
        // Callback to handle changing the case
        //
        d3.select ('#caseList').on ('change', function ()
        {
            selectedCase = this.value;

            // console.log ('caseList... ', selectedCase);

            UpdateLabels ();
            UpdateImages ();
            UpdateGraph ();

        } );

    } ); // d3.csv

} // D3App



