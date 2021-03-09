function D3App ()
{
    // The data...
    var dataFile = 'BinaryClassifier_TotalResults.csv';

    var dataset, xScale, yScale, xAxis, yAxis;

    var currDataType = 'TP';

    var selected = 0;
    var selectedRange = 1;

    var colours = [ '#006699', '#669900', '#990066', '#cc6600' ];
    var barLabels = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ];
    var dataTypeIndex = { 'TP' : 2, 'TN' : 3, 'FP' : 4, 'FN': 5, 'P': 6, 'R': 7, 'F1': 8 };

    var classColourIndex = { 'SGD' : 0, 'RF': 1, 'LOG': 2, 'KN': 3 };
    var classVisState = { 'SGD' : true, 'RF': true, 'LOG': true, 'KN': true }; 
    var classIndex = [ 'SGD', 'RF', 'LOG', 'KN' ];

    var showToolTips = false;
    var toolTipLabels = [ 'Stochastic Gradient Descent', 'Random Forest', 'Logistic Regression', 'K-Neighbours' ];

 
    // SVG Width, height, and some added spacing
    var margin = {
            top:    5,
            right:  5,
            bottom: 20,
            left:   40
    };

    var width  = 620 - margin.left - margin.right;
    var height = 400 - margin.top - margin.bottom;

    var updateDuration = 500;


    //
    // Function used to parse the CSV.  
    // 
    var rowConverter = function (d) 
    {
        //console.log (d);

        return [
            d['ClassiferType'],          // 0
            parseInt (d['Number']),      // 1
            parseInt (d['numTruePos']),  // 2
            parseInt (d['numTrueNeg']),  // 3
            parseInt (d['numFalsePos']), // 4
            parseInt (d['numFalseNeg']), // 5
            parseFloat (d['precision']), // 6
            parseFloat (d['recall']),    // 7
            parseFloat (d['F1Score'])    // 8
        ];  
    }


    //
    // Read the CSV...
    //
    d3.csv (dataFile, rowConverter).then (function (data) 
    {
        // console.log (data);            
        dataset = data;

        // Find various Max values for the true/false positive/negatives 
        // this is just used to scale our graphs appropriately.  Note that
        // precision, recall, and F1 score have a max value of 1
        var maxValues = [ 0, 0, 0, 0, 1, 1, 1 ];
        for (var j = 0; j < 4; j++)
        {
            var tmpValues = []
            for (var i = 0; i < dataset.length; i++)
            {
                tmpValues.push (dataset[i][j+2]);
            }
            maxValues [j] = d3.max (tmpValues);
        }

        // uncomment for debugging...
        // console.log (maxValues);
 
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
                   .domain ( [ 0, maxValues[selected] ] )
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

        var space    = xScale(1.03) - xScale(1);
        var barWidth = xScale(1.235) - xScale(1);

        d3.select ('#btn_SGD').style ('background-color', colours[0]);
        d3.select ('#btn_SGD').style ('color', '#fff');
        d3.select ('#btn_RF').style ('background-color', colours[1]);
        d3.select ('#btn_RF').style ('color', '#fff');
        d3.select ('#btn_LOG').style ('background-color', colours[2]);
        d3.select ('#btn_LOG').style ('color', '#fff');
        d3.select ('#btn_KN').style ('background-color', colours[3]);
        d3.select ('#btn_KN').style ('color', '#fff');


        for (var j = 0; j < 4; j++)
        {
            for (var i = 0; i < 10; i++)
            {
                var xValue = 1.25*space + xScale(i + 0.235*j);
                var yValue = yScale (dataset[i + 10*j][selected+2]);

                svg.append ('rect')
                   .attr ('id',     ('rect_' + j + '_' + i) )
                   .attr ('x',      xValue )
                   .attr ('y',      yValue )
                   .attr ('width',  barWidth )
                   .attr ('height', yScale(0) - yValue )
                   .attr ('fill',   colours [j] )
                   .attr ('tooltipText', toolTipLabels[j] + '<hr>' + dataset[i + 10*j][selected+2] )
                   .on ('mouseover', function (d)
                   {
                       if (showToolTips == false)
                           return;

                       var xPosition = parseFloat (d3.select(this).attr('x')) + (xScale(1) - xScale(0)) / 2;
                       var yPosition = parseFloat (d3.select(this).attr('y')) + Math.abs(yScale(0) - yScale(d)) / 2;

                       //console.log (d3.mouse(this));
                       xPosition = d3.event.pageX; //d3.mouse(this)[0];
                       yPosition = d3.mouse(this)[1];

                       //console.log (xPosition + '   ' + yPosition);
                       d3.select ('#tooltip')
                         .style ('left', xPosition + 'px')
                         .style ('top', yPosition + 'px')
                         .select ('#label').html ( d3.select (this).attr ('tooltipText') );

                       d3.select ('#tooltip').classed ('hidden', false);
                   })
                   .on ('mouseout', function (d)
                   {
                       d3.select ('#tooltip').classed ('hidden', true);
                   })
                 
            }
        }


        //
        // Used to redraw the graph
        //
        //var UpdateGraph = function (selData, selRange)
        function UpdateGraph ()
        {
            // update the y-axis
            yScale.domain ( [ 0, maxValues[ dataTypeIndex[currDataType] - 2 ] ] );

            svg.select ('.y.axis')
               .transition ()
               .duration (updateDuration)
               .call (yAxis);

            // find the number of bars
            var numBars = 0;
            for (var key in classVisState)
                numBars += classVisState [key];

            var totalWidth = (xScale(2) - xScale(1)) - 2*space;
            barWidth =  totalWidth / numBars;

            var xStart = 1.25*space;
            for (var j = 0; j < 4; j++) 
            {  
                var drawCurrBar = classVisState[classIndex[j]];
                var currBarWidth = 0;
                if ( drawCurrBar == true )
                {
                    currBarWidth = barWidth;
                }

                for (var i = 0; i < 10; i++)
                { 
                    var yValue = yScale (dataset[i + 10*j][ dataTypeIndex[currDataType] ]);
                   
                    d3.select ('#rect_' + j + '_' + i) 
                      .transition ()
                      .duration (updateDuration)
                      .attr ('x',      xStart + xScale(i))
                      .attr ('y',      yValue )
                      .attr ('width',  currBarWidth )
                      .attr ('height', (yScale(0) - yValue) )
                      .attr ('tooltipText', toolTipLabels[j] + '<hr>' + dataset[i + 10*j][ dataTypeIndex[currDataType] ] );
                }

                if ( drawCurrBar == true )
                    xStart += barWidth;
            }

        }  // update graph function


        //
        // Handle the various datatype buttons
        //
        d3.selectAll ('.datatype').on ('click', function()
        {
            var selDataType = d3.select(this).node().getAttribute ('dataType');

            var selBtnID = -1;
            if (selDataType != currDataType)
            {
                // update the UI
                d3.select ('#btn_' + currDataType).classed ('button_sel', false);
                d3.select ('#btn_' + selDataType).classed ('button_sel', true); 
 
                currDataType = selDataType;
            }
            else 
            {
                // do nothing...
                return;
            } 

            UpdateGraph ();

        } );


        //
        // Handle the the absolute/percent buttons
        //
        d3.selectAll ('.classType').on ('click', function()
        {
            //console.log ('classType button CB');  

            var selClassType = d3.select(this).node().getAttribute ('classType');

            // update the button
            if ( classVisState[selClassType] == true )
            {
                d3.select ('#btn_' + selClassType).style ('background-color', '#ccc');
                d3.select ('#btn_' + selClassType).style ('color', '#000');

                classVisState[selClassType] = false;
            }
            else
            {
                d3.select ('#btn_' + selClassType).style ('background-color', colours [ classColourIndex[selClassType] ]);
                d3.select ('#btn_' + selClassType).style ('color', '#fff');

                classVisState[selClassType] = true;
            }

            UpdateGraph ();
                        
        } ); 


        //
        // Hide/Show tooltip callback
        //
        d3.select ('#tooltipsCB').on ('click', function(d)
        {
            showToolTips = !showToolTips;
        } );

    } ); // d3.csv

} // D3App


