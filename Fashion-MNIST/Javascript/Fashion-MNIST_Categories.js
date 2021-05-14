function ImgDisplayApp ()
{
    //
    // Callback to handle the clicks...
    //
    function labelUpdateClick ()
    {
        // console.log ('labelUpdateClick');

        var selected = this;
        if (selected != currentSelected)
        {
            selected.style.background = '#002233';
            selected.style.color = '#fff';
            
            currentSelected.style.background = '#ccc';
            currentSelected.style.color = '#000';

            var imageStr = this.getAttribute ('image');
            //console.log (imageStr);

            document.getElementById ('ImageBox').src = imageStr;

            currentSelected = selected;
        }
    }


    //
    // Callback to handle mouse leaving a label
    //
    function labelUpdateMouseEnter ()
    {
        // console.log ('labelUpdateMouseEnter');

        var selected = this;
        if (selected != currentSelected)
        {
            selected.style.background = '#e62e00';
            selected.style.color = '#fff';
        } 
    }

    //
    // Callback to handle mouse entering a label
    //
    function labelUpdateMouseLeave ()
    {
        // console.log ('labelUpdateMouseLeave');

        var selected = this;
        if (selected != currentSelected)
        {
            selected.style.background = '#ccc';
            selected.style.color = '#000';
        } 
    }


    // add callbacks for all the labels...
    var elements = document.getElementsByClassName ('label');
    var currentSelected = elements[0];
    for (var i = 0; i < elements.length; i++) 
    {
        elements[i].addEventListener ('click', labelUpdateClick, false);
        elements[i].addEventListener ('mouseenter', labelUpdateMouseEnter, false);
        elements[i].addEventListener ('mouseleave', labelUpdateMouseLeave, false);
    }
}


