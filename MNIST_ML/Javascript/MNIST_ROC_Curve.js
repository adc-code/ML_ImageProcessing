function D3App ()
{
    // SVG Width, height, and some added spacing
    var margin = { top: 10, right: 30, bottom: 40, left: 65 };
    var width  = 480 - margin.left - margin.right;
    var height = 480 - margin.top  - margin.bottom;
    
    var selectedNum = 0;
    var Colours = [ '#006699', '#669900', '#990066', '#cc6600' ];

    var btnLabels = [ 'Stochastic Gradient Descent <hr> AUC: ',
                      'Random Forest Classifier <hr> AUC: ',
                      'Logistic Regression <hr> AUC: ',
                      'K-Neightbours Classifier <hr> AUC: ' ];
    var btnIDs = [ '#SGDBtn', '#RFBtn', '#LogBtn', '#KNBtn' ];

    var updateDuration = 500;

    var visState = [ true, true, true, true ];

    var dataset = -1;

    // Create SVG element
    var svg = d3.select ('#graph')
                .append ('svg')
                .attr ('width', width + margin.left + margin.right)
                .attr ('height', height + margin.top + margin.bottom)
                .append ('g')
                .attr ('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    // Define the scales to convert our data to screen coordinates
    xScale = d3.scaleLinear ()
               .domain ( [ -0.01, 1.01 ] )
               .range ( [ 0, width ] );                

    yScale = d3.scaleLinear ()
               .domain ( [ -0.01, 1.01 ] )
               .range ( [ height, 0 ] );

    // Define the xAxis... note only a line is shown
    var xAxis = d3.axisBottom ()
                  .tickSize ( 0 )
                  .scale (xScale)
                  .tickFormat ('');

    // Define the vertical lines for the grid using very long tick lines
    var xGrid = d3.axisBottom ()
                  .tickSize ( -height )
                  .scale (xScale);

    // Define Y axis
    var yAxis = d3.axisLeft ()
                  .scale (yScale);
                          
    // Define the horizontal lines for the grid again using very long tick lines
    var yGrid = d3.axisLeft ()
                  .tickSize (-width, 0, 0)
                  .scale (yScale)
                  .tickFormat ('');

    svg.append ('clipPath')
       .attr ('id', 'graphArea')
       .append ('rect')
       .attr ('x', -1)
       .attr ('y', -1)
       .attr ('width', width+2)
       .attr ('height', height+2);

    // Create axes
    svg.append ('g')
       .attr ('class', 'x axis')
       .attr ('transform', 'translate(0,' + yScale(-20) + ')')
       .call (xAxis);

    svg.append ('g')
       .attr ('class', 'grid')
       .attr ('transform', 'translate(0,' + height + ')')
       .call (xGrid);

    svg.append ('g')
       .attr ('class', 'y axis')
       .attr ('transform', 'translate(' + 0 + ',0)')
       .call (yAxis); 
 
    svg.append ('g')
       .attr ('class', 'grid')
       .attr ('transform', 'translate(' + 0 + ',0)')
       .call (yGrid); 

    svg.append ('text')
       .style ('text-anchor', 'middle')
       .attr ('transform', 'rotate(-90)')
       .attr ('x', -height * 0.5)
       .attr ('y', -margin.left * 0.67)
       .text ('True Positive Rate / Sensitivity / Recall')
       .attr ('class', 'label');

    svg.append ('text')
       .style ('text-anchor', 'middle')
       .attr ('x', xScale(0.5))
       .attr ('y', yScale(-0.1) )
       .text ('False Positive Rate / Specificity / Selectivity')
       .attr ('class', 'label');


    // Define the lines...
    var horizLine = d3.line ()
                      .x (function(d) { return d.x; })
                      .y (function(d) { return d.y; });
 
    var vertLine = d3.line ()
                     .x (function(d) { return d.x;  })
                     .y (function(d) { return d.y;  });

    var solutionLine = [];
    for (var i = 0; i < 40; i++)
    {
        solutionLine.push (d3.line ()
                    .x (function(d) { return xScale (d.x);  })
                    .y (function(d) { return yScale (d.y);  })
                    .curve (d3.curveLinear) );
    }


    // Add the query lines to the svg
    svg.append ('path')
       .datum ( [ {x: xScale.range()[0], y: yScale(0)}, {x: xScale.range()[1], y: yScale(0)} ] )
       .attr ('class', 'queryLine') 
       .attr ('id', 'horizLine')
       .attr ('d', horizLine);

    svg.append ('path')
       .datum ( [ {x: xScale(0), y: yScale.range()[0]}, {x: xScale(0), y: yScale.range()[1]} ] )
       .attr ('class', 'queryLine') 
       .attr ('id', 'vertLine')
       .attr ('d', vertLine);

    // Make an empty rectangle to handle mouse move events...
    svg.append ('rect')
       .attr ('width', width)
       .attr ('height', height)
       .style ('fill', 'none')
       .style ('pointer-events', 'all')
       .on ('mousemove', mouseMove);

    var nullLine = [ { x: xScale.range()[0], y: 0.5 * (yScale.range()[0] + yScale.range()[1]) }, 
                     { x: xScale.range()[1], y: 0.5 * (yScale.range()[0] + yScale.range()[1]) } ];


    var solPath = [];
    for (var i = 0; i < 40; i++)
    {
        solPath = svg.append ('path')
                     .datum (nullLine)         
                     .attr ('class', 'solutionLine')
                     .attr ('id', 'solutionLine_' + i)
                     .attr ('d', solutionLine[i])
                     .attr ('clip-path', 'url(#graphArea)')
                     .style ('opacity', 0)
                     .style ('stroke', Colours [ Math.floor (i/10) ] );
    }


    // UI related
    for (var i = 0; i < 4; i++)
    {
        d3.select (btnIDs[i]).node().style.backgroundColor = Colours [i];
        d3.select (btnIDs[i]).node().style.color = '#fff';
    }

    d3.json ('BinaryClassifier_TotalROCCurvess.json')
      .then ( function (data) 
        {
            dataset = data;

            for (var j = 0; j < 4; j++)
            {
                var num = 10*j;

                // recompute the curve
                var ROCCurveData = [];
                for (var i = 0; i < data[num].TPR.length; i++)
                {
                    ROCCurveData.push ( { x: data[num].FPR[i], y: data[num].TPR[i] } );
                }
             
                // redraw the curve
                svg.select ('#solutionLine_' + num)
                   .style ('opacity', 1)
                   .attr ('d', solutionLine[num] (ROCCurveData) ); 

                d3.select(btnIDs[j]).html (btnLabels[j] + data[num].AUC.toFixed(8))
            }

            d3.selectAll ('.cell').on ('click', function ()
            {
                var val = +d3.select(this).node().getAttribute ('val');

                if (val != selectedNum)
                {
                    d3.selectAll ('.cell').classed ('cellSel', false);
                    d3.select (this).classed ('cellSel', true);

                    for (var j = 0; j < 4; j++)
                    {
                        var num = val + 10*j;
                        var prevNum = selectedNum + 10*j;

                        var ROCCurveData = [];
                        for (var i = 0; i < data[num].TPR.length; i++)
                        {
                            ROCCurveData.push ( { x: data[num].FPR[i], y: data[num].TPR[i] } );
                        }
             
                        // redraw the curve
                        svg.select ('#solutionLine_' + num)
                           .attr ('d', solutionLine[num] (ROCCurveData) );

                        if ( visState [j] == true )
                        {
                            svg.select ('#solutionLine_' + prevNum)
                               .transition ()
                               .duration (updateDuration) 
                               .style ('opacity', 0);

                            svg.select ('#solutionLine_' + num)
                               .transition ()
                               .duration (updateDuration) 
                               .style ('opacity', 1);

                            d3.select(btnIDs[j]).html (btnLabels[j] + data[num].AUC.toFixed(8));   
                        }
                    }
                }

                selectedNum = val;
                     
            } );

        } );


    //
    // mousemove callback function used for the query line
    //
    function mouseMove ()
    {
        var selectedX = d3.mouse (this)[0];
        var selectedY = d3.mouse (this)[1];

        // console.log (selectedX, selectedY);
 
        var locData = [ { x: xScale.range()[0], y: selectedY }, 
                        { x: xScale.range()[1], y: selectedY } ];
        svg.select ('#horizLine').attr ('d', horizLine (locData));

        locData = [ { x: selectedX, y: yScale.range()[0] }, 
                    { x: selectedX, y: yScale.range()[1] } ];
        svg.select ('#vertLine').attr ('d', vertLine (locData));
    }


    // 
    // callback to handle click on the reseet button
    //
    d3.selectAll ('.button').on ('click', function ()
    {
        var selBtn = '#' + d3.select (this).node().getAttribute ('id');
        var visId  = d3.select (this).node().getAttribute ('visId');
        
        // change UI state
        if ( visState [visId] == true )
        {
            d3.select (selBtn).node().style.backgroundColor = '#ccc';
            d3.select (selBtn).node().style.color = '#000';
                    
            visState [visId] = false;

            svg.select ('#solutionLine_' + (selectedNum + 10*visId))
               .transition ()
               .duration (updateDuration)
               .style ('opacity', 0);

            d3.select (selBtn).html (btnLabels[visId] + ' - ');
        }
        else
        {
            d3.select (selBtn).node().style.backgroundColor = Colours [visId];
            d3.select (selBtn).node().style.color = '#fff';
                    
            visState [visId] = true;

            svg.select ('#solutionLine_' + (selectedNum + 10*visId))
               .transition ()
               .duration (updateDuration)
               .style ('opacity', 1);

            d3.select (selBtn).html (btnLabels[visId] + dataset[(selectedNum + 10*visId)].AUC.toFixed(8));
        }

    } );

}



