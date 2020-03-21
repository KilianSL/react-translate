import React, { Component } from 'react'
import './TranslateBox.css'

class TranslateBox extends Component{
    
    constructor(props){
        super(props)
        this.state = {
            input_text: ""
        }
    }

    render(){
        return(
            <div className="content-container">
                <div id="input" className="translate-container">
                    <div className="translate-header">
                        <span>German</span>
                    </div>
                    <div className="translate-textbox">
                        <textarea placeholder="Enter Text..." />
                    </div>
                </div>
                <div id="output" className="translate-container">
                    <div className="translate-header">
                        <span>English</span>
                    </div>
                    <div className="translate-textbox">
                        <textarea placeholder="Translation" readOnly/>
                    </div>
                </div>
            </div>
        )
    }
}

export default TranslateBox