import React, { Component } from 'react'
import './TranslateBox.css'

const API_IP = "http://127.0.0.1:5000/?src="

class TranslateBox extends Component{
    
    constructor(props){
        super(props)
        this.state = {
            input_text: "",
            output_text: ""
        }
    }


    handleTextUpdate = (e) => {
        var src = e.target.value
            fetch(API_IP + src)
            .then(request => request.text())
            .then(translated => this.setState({
            input_text : src,
            output_text : translated
            }, console.log(this.state.output_text)))
    }

    render(){
        return(
            <div className="content-container">
                <div id="input" className="translate-container">
                    <div className="translate-header">
                        <span>German</span>
                    </div>
                    <div className="translate-textbox">
                        <textarea placeholder="Enter Text..." onChange={this.handleTextUpdate} />
                    </div>
                </div>
                <div id="output" className="translate-container">
                    <div className="translate-header">
                        <span>English</span>
                    </div>
                    <div className="translate-textbox">
                        <textarea placeholder="Translation" value={this.state.output_text} readOnly></textarea>
                    </div>
                </div>
            </div>
        )
    }
}

export default TranslateBox