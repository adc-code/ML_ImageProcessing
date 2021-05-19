function D3App ()
{
    var dataFileName = 'FashionMNIST-ConfMatData.json';

    // SVG Width, height, and some added spacing
    var margin = {
            top:    0,
            right:  0,
            bottom: 85,
            left:   70
    };

    var predGraphMargin = {
            top:    0,
            right:  50,
            left:   70,
            bottom: 0
    };

    var width  = 450 - margin.left - margin.right;
    var height = 450 - margin.top - margin.bottom;

    var graphWidth  = 245 - predGraphMargin.left - predGraphMargin.right;
    var graphHeight = 260 - predGraphMargin.top - predGraphMargin.bottom;

    var labels = [ 'T-shirt-Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'AnkleBoot' ];

    const updateDuration = 500;
    const DARKNESSCOEF   = 20;

    var maxOffDiagValue = 0;

    var selectedImage = 0;
    var selectedElem  = 0;
    var selectedBox   = null;

    const hoverColour   = '#ff66d9';
    const selectColour  = '#cc0099';
    const defaultColour = '#000000';

    const defaultWidth  = 1;
    const selectWidth   = 4;


    d3.json (dataFileName).then (function (data) 
    {
        var filterData = data;
        console.log (data);

        // figure out the off diagonal max value
        for (var i = 0; i < filterData.length; i++)
        {
            if (i % 11 == 0)
                continue;

            if (filterData[i]['Count'] > maxOffDiagValue)
                maxOffDiagValue = filterData[i]['Count'];
        }

        var svg = d3.select ('#confusionMatrixArea').append('svg')
                    .attr ('width', width + margin.left + margin.right)
                    .attr ('height', height + margin.top + margin.bottom)
                    .append ('g')
                    .attr ('transform', 'translate(' + margin.left + ',' + margin.top + ')');
	
        // Define the scales to convert our data to screen coordinates
        var xScale = d3.scaleLinear ()
                       .domain ( [ 0, 10 ] )
                       .range ( [ 0, width ] ); 

        var yScale = d3.scaleLinear ()
                       .domain ( [ 0, 10 ] )
                       .range ( [ height, 0 ] ); 

        var colourScaleStd = d3.scaleLinear ()
                               .domain ( [ 0, 1000 ] )
                               .range ( [ 0.05, 0.55 ] );

        var colourScaleAdj = d3.scaleLinear ()
                               .domain ( [ 0, 1 ] )
                               .range ( [ 0.05, 0.55 ] );

        var xScalePredGraph = d3.scaleLinear ()
                                .domain ( [ 0, 100 ] )
                                .range ( [ 1, graphWidth ] ); 

        var yScalePredGraph = d3.scaleLinear ()
                                .domain ( [ 0, 10 ] )
                                .range ( [ graphHeight, 0 ] );

        var svgPredGraph = d3.select ('#predGraph')
                             .append ('svg')
                             .attr ('width', graphWidth + predGraphMargin.left + predGraphMargin.right)
                             .attr ('height', graphHeight + predGraphMargin.top + predGraphMargin.bottom)
                             .style ('background-color', 'rgb(247, 241, 247)')
                             .append ('g')
                             .attr ('transform', 'translate(' + predGraphMargin.left + ',' + predGraphMargin.top + ')');

        var xAxisPredGraph = d3.axisBottom ()
                               .scale (xScalePredGraph)
                               .ticks (10)
                               .tickSizeInner ( 10 )
                               .tickSizeOuter ( 10 )
                               .tickPadding ( -1 );

        // Draw bars and text for the post titles and view counts
        var barHeight = yScale(1) - yScale(2);
        for (var i = 0; i < 10; i++)
        {
            for (var j = 0; j < 10; j++)
            {
                var box = svg.append ('g')
                             .attr ('class', 'box');

                box.append ('rect')
                   .attr ('x', xScale (i)+2)
                   .attr ('y', yScale (10 - j)+2 )
                   .attr ('width', xScale (1)-4)
                   .attr ('height', barHeight-4)
                   .attr ('stroke', defaultColour)
                   .attr ('stroke-width', defaultWidth)
                   .attr ('id', 'box_' + i + '_' + j);

                box.append ('text')
                   .attr ('x', xScale (i + 0.5))
                   .attr ('y', yScale (10 - j - 0.55))
                   .attr ('text-anchor', 'middle')
                   .attr ('alignment-baseline', 'middle')
                   .attr ('id', 'text_' + i + '_' + j)
                   .attr ('class', 'label')
                   .text ('-');
            }

            svg.append ('text')
               .attr ('x', xScale (-0.1))
               .attr ('y', yScale (10 - i - 0.55))
               .attr ('text-anchor', 'end')
               .attr ('alignment-baseline', 'middle')
               .attr ('class', 'label')
               .text (labels [i]);

            svg.append ('text')
               .attr ('x', 0)
               .attr ('y', 0)
               .attr ('text-anchor', 'start')
               .attr ('class', 'label')
               .attr ('transform', 'translate(' + xScale (i+0.4) + ',' + yScale (-0.15) + ')  rotate(90)')
               .text (labels [i]);
        }


        //
        // Manually add the axis labels...
        //
        svg.append ('text')
           .attr ('x', xScale (5))
           .attr ('y', yScale (-2))
           .attr ('text-anchor', 'middle')
           .attr ('class', 'title')
           .text ('Predicted Results');

        svg.append ('text')
           .attr ('x', 0)
           .attr ('y', 0)
           .attr ('text-anchor', 'middle')
           .attr ('class', 'title')
           .attr ('transform', 'translate( ' + xScale(-1.6) + ',' + yScale (5) + ') rotate(270)') 
           .text ('Actual Item');


        //
        // Make the bar graph which contains the prediction precentages...
        //
        for (var i = 0; i < 10; i++)
        {
            svgPredGraph.append ('rect')
                        .attr ('id',     ('pred_' + i) )
                        .attr ('x',      xScalePredGraph (0) )
                        .attr ('y',      yScalePredGraph (10 - i) + 2)
                        .attr ('width',  xScalePredGraph (0) )
                        .attr ('height', yScalePredGraph (0) - yScalePredGraph (1) - 4 )
                        .attr ('fill',   d3.interpolatePuBu (0.85) )

            svgPredGraph.append ('text')
                        .attr ('x', -5)
                        .attr ('y', yScalePredGraph(10 - i - 0.65))
                        .attr ('text-anchor', 'end')
                        .attr ('class', 'label')
                        .text (labels [i]);

            svgPredGraph.append ('text')
                        .attr ('id', ('predPct_' + i) )
                        .attr ('x', xScalePredGraph (0) + 5)
                        .attr ('y', yScalePredGraph (10 - i - 0.65))
                        .attr ('text-anchor', 'start')
                        .attr ('class', 'label')
                        .text ('0%');
        }


        // 
        // Update all the values in the confusion matrix
        //
        for (var elem = 0; elem < filterData.length; elem++)
        {
            var text = d3.select ('#text_' + filterData[elem].j + '_' + filterData[elem].i);
            text.text (filterData[elem].Count);
        }

            
        //
        // Manually set some defaults for the UI
        //
        selectedBox = d3.select('.box');
        var firstBoxId = selectedBox.select ('rect').attr ('id');
        var idTokens = firstBoxId.split ('_');
        selectedElem = parseInt (idTokens[2]) * 10 + parseInt (idTokens[1]);
        selectedBox.select('rect').attr ('stroke', selectColour);
        selectedBox.select('rect').attr ('stroke-width', selectWidth);
        UpdateSelection ();

        // set default colour scheme button
        d3.select ('#btn_0').node().style.backgroundColor = d3.interpolatePuBu (0.85);
        d3.select ('#btn_0').node().style.color = '#fff';

        // set box colours
        UpdateBoxColours (0);


        //
        // Used to change the shading of the box colours
        //
        function UpdateBoxColours (style)
        {
            if (style == 0)
            {
                for (var elem = 0; elem < filterData.length; elem++)
                {
                    d3.select ('#box_' + filterData[elem].j + '_' + filterData[elem].i)
                      .transition ()
                      .duration (updateDuration)
                      .attr ('fill', d3.interpolatePuBu (colourScaleStd (filterData[elem].Count)) );
                }
            }
            else if (style == 1)
            {
                for (var elem = 0; elem < filterData.length; elem++)
                {
                    var colour = '#ffffff';

                    if (elem % 11 != 0)
                    {
                        var value = Math.log (DARKNESSCOEF * filterData[elem].Count + 1) / Math.log (DARKNESSCOEF * maxOffDiagValue + 1);
                        colour = d3.interpolatePuBu ( colourScaleAdj ( value ) );
                    }
 
                    d3.select ('#box_' + filterData[elem].j + '_' + filterData[elem].i)
                      .transition ()
                      .duration (updateDuration)
                      .attr ('fill', colour );
                }
            }
        }


        //
        // Used to update the prediction graph
        //
        function UpdateGraph ()
        {
            for (var i = 0; i < 10; i++)
            {
                var value = 0;
                if (filterData[selectedElem]['PredictionValues'].length > 0)
                    value = filterData[selectedElem]['PredictionValues'][selectedImage][i];

                svgPredGraph.select ('#pred_' + i)
                            .transition ()
                            .duration (updateDuration)
                            .attr ('width', xScalePredGraph (value) ); 

                svgPredGraph.select ('#predPct_' + i)
                            .transition ()
                            .duration (updateDuration)
                            .attr ('x', xScalePredGraph (value) + xScalePredGraph(1)) 
                            .text (value + '%'); 
            }
        }


        //
        // Used to update images and the prediction graph after a selection was made 
        //
        function UpdateSelection ()
        {
            // update images...
            for (var i = 0; i < 5; i++)
            {
                var imageName = "FashionMNIST_OutputImages/FashionMNIST_Item_nothing.png";
                var newState  = 0;
                if (i < filterData[selectedElem]['IDs'].length)
                {
                    imageName = "FashionMNIST_OutputImages/FashionMNIST_Item_" + filterData[selectedElem]['IDs'][i] + ".png";
                    newState  = 1;
                }

                document.getElementById ('Image_' + i).src = imageName;
                document.getElementById ('ImageBox_' + i).setAttribute ('state', newState);
                document.getElementById ('ImageBox_' + i).style.backgroundColor = defaultColour;
            }

            // update the label
            var msg = 'No images to show';
            if (filterData[selectedElem]['Count'] == 1)
                msg = 'Showing only 1 image'; 
            if (filterData[selectedElem]['Count'] > 1 && filterData[selectedElem]['Count'] <= 5)
                msg = 'Showing all ' + filterData[selectedElem]['Count'] + ' images';
            else if (filterData[selectedElem]['Count'] > 5)
                msg = 'Showing first 5 images (of a total of ' + filterData[selectedElem]['Count'] + ')'; 
            document.getElementById('imageLabel').innerText = msg;

            // make the first image selected, if more than one exists
            if (filterData[selectedElem]['Count'] > 0)
                document.getElementById ('ImageBox_0').style.backgroundColor = selectColour;

            // update the prediction graph
            UpdateGraph (); 
        }


        //
        // callback to handle clicking on a confusion matrix box element
        // 
        d3.selectAll ('.box').on ('click', function ()
        {
            var id = d3.select (this).select ('rect').attr ('id');

            var idTokens = id.split ('_');
            var currElem = parseInt (idTokens[2]) * 10 + parseInt (idTokens[1]);

            if (currElem != selectedElem)
            {
                // update UI...
                // make the previous selected box normal/unselected
                selectedBox.select('rect').attr ('stroke', defaultColour);
                selectedBox.select('rect').attr ('stroke-width', defaultWidth);
           
                // colour the currently selected box... that is select it.
                d3.select(this).select('rect').attr ('stroke', selectColour);
                d3.select(this).select('rect').attr ('stroke-width', selectWidth);

                // keep track of the selected state
                selectedBox  = d3.select (this);
                selectedElem = currElem;

                selectedImage = 0;
                UpdateSelection ();
            }

        } );


        //
        // callback to handle when the mouse enters a confusion matrix box element
        // 
        d3.selectAll ('.box').on ('mouseenter', function ()
        {
            d3.select(this).select('rect').attr ('stroke', hoverColour)
            d3.select(this).select('rect').attr ('stroke-width', 4)
        } );


        //
        // callback to handle when the mouse leaves a confusion matrix box element
        // 
        d3.selectAll ('.box').on ('mouseleave', function ()
        {
            var id = d3.select (this).select ('rect').attr ('id');

            var idTokens = id.split ('_');
            var currElem = parseInt (idTokens[2]) * 10 + parseInt (idTokens[1]);

            var strokeColour = defaultColour;
            var strokeWidth  = defaultWidth;

            if (currElem == selectedElem)
            {
                strokeColour = selectColour;
                strokeWidth  = selectWidth;
            }

            d3.select(this).select('rect').attr ('stroke', strokeColour);
            d3.select(this).select('rect').attr ('stroke-width', strokeWidth);
        } );


        //
        // callback to handle when the mouse enters an image
        // 
        d3.selectAll ('.image').on ('mouseenter', function ()
        {
            var newColour = defaultColour;
            if (this.getAttribute('state') == 1)
                newColour = hoverColour;

            d3.select(this).node().style.backgroundColor = newColour;
        } );


        //
        // callback to handle when the mouse leaves an image
        // 
        d3.selectAll ('.image').on ('mouseleave', function ()
        {
            var idName = d3.select(this).attr ('id');
            var idTokens = idName.split ('_');
            var currSelectedImage = parseInt (idTokens[1]);

            var newColour = defaultColour;
            if (currSelectedImage == selectedImage && this.getAttribute('state') == 1)
                newColour = selectColour;

            d3.select(this).node().style.backgroundColor = newColour;
        } );


        //
        // callback to handle clicking on an image
        // 
        d3.selectAll ('.image').on ('click', function ()
        {
            if (this.getAttribute('state') == 0)
                return;

            var idName = d3.select(this).attr ('id');
            var idTokens = idName.split ('_');

            var currSelectedImage = parseInt (idTokens[1]);

            if (currSelectedImage != selectedImage)
            {
                d3.select('#ImageBox_' + selectedImage).node().style.backgroundColor = defaultColour;
                d3.select('#ImageBox_' + currSelectedImage).node().style.backgroundColor = selectColour;

                selectedImage = currSelectedImage;
            }

            UpdateGraph ();
        } );


        //
        // callback to handle clicking on either shading style button 
        //
        d3.selectAll ('.button').on ('click', function ()
        {
            var idName = d3.select(this).attr ('id');
            var idTokens = idName.split ('_');

            // update the UI... perhaps this isn't the best way of doing this but we only have two states
            var focusedId   = '#btn_0';
            var unfocusedId = '#btn_1';

            if (idName == 'btn_1')
            {
                focusedId   = '#btn_1';
                unfocusedId = '#btn_0';
            }
                    
            d3.select (focusedId).node().style.backgroundColor = d3.interpolatePuBu (0.85);
            d3.select (focusedId).node().style.color = '#fff';
            d3.select (unfocusedId).node().style.backgroundColor = '#fff';
            d3.select (unfocusedId).node().style.color = '#000';
           
            UpdateBoxColours ( parseInt (idTokens[1]) ); 
        } );

    } );

}


